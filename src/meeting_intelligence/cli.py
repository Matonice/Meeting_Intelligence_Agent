"""CLI entry point for the Meeting Intelligence Agent."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import load_config
from .models.meeting import MeetingResult
from .models.transcript import Transcript
from .pipeline import Pipeline
from .understanding.llm_client import LLMClient
from .understanding.qa import MeetingQA
from .utils.logging import setup_logging

console = Console()


@click.group()
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="Path to config.yaml")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx: click.Context, config_path: str | None, verbose: bool) -> None:
    """Meeting Intelligence Agent - Turn meeting recordings into structured knowledge."""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(Path(config_path) if config_path else None)
    ctx.obj["verbose"] = verbose


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--output-dir", "-o", type=click.Path(), default="./output",
              help="Output directory")
@click.option("--speakers", "-s", type=int, default=None,
              help="Expected number of speakers")
@click.option("--language", "-l", type=str, default=None,
              help="Language code (e.g., en, fr, de)")
@click.option("--no-llm", is_flag=True,
              help="Skip GPT-4 analysis (transcript only)")
@click.option("--tts", is_flag=True,
              help="Generate audio summary playback")
@click.option("--question", "-q", type=str, multiple=True,
              help="Questions to ask about the meeting (repeatable)")
@click.pass_context
def process(
    ctx: click.Context,
    audio_file: str,
    output_dir: str,
    speakers: int | None,
    language: str | None,
    no_llm: bool,
    tts: bool,
    question: tuple[str, ...],
) -> None:
    """Process a meeting recording through the full pipeline."""
    config = ctx.obj["config"]
    config.output.directory = output_dir

    console.print(Panel(f"[bold]Processing:[/] {audio_file}", style="blue"))

    pipeline = Pipeline(config)
    result = pipeline.process(
        audio_path=Path(audio_file),
        num_speakers=speakers,
        language=language,
        skip_llm=no_llm,
        enable_tts=tts,
        questions=list(question) if question else None,
    )

    _display_result(result)


@cli.command()
@click.argument("transcript_file", type=click.Path(exists=True))
@click.option("--question", "-q", type=str, required=True,
              help="Question to ask about the meeting")
@click.pass_context
def ask(ctx: click.Context, transcript_file: str, question: str) -> None:
    """Ask a question about a previously processed meeting."""
    config = ctx.obj["config"]

    # Load transcript from JSON
    raw = json.loads(Path(transcript_file).read_text())

    # Handle both full MeetingResult and standalone Transcript JSON
    if "transcript" in raw:
        transcript = Transcript.model_validate(raw["transcript"])
    else:
        transcript = Transcript.model_validate(raw)

    llm = LLMClient(config.llm)
    qa = MeetingQA(llm)
    response = qa.ask(question, transcript)

    console.print()
    console.print(f"[bold]Q:[/] {response.question}")
    console.print(f"[bold green]A:[/] {response.answer}")
    console.print(f"[dim]Confidence: {response.confidence:.0%}[/]")


@cli.command()
@click.argument("transcript_file", type=click.Path(exists=True))
@click.pass_context
def summarize(ctx: click.Context, transcript_file: str) -> None:
    """Re-generate summary for an existing transcript."""
    config = ctx.obj["config"]

    raw = json.loads(Path(transcript_file).read_text())
    if "transcript" in raw:
        transcript = Transcript.model_validate(raw["transcript"])
    else:
        transcript = Transcript.model_validate(raw)

    from .understanding.summarizer import MeetingSummarizer
    from .understanding.extractor import MeetingExtractor

    llm = LLMClient(config.llm)
    extractor = MeetingExtractor(llm)
    summarizer = MeetingSummarizer(llm)

    console.print("[bold]Analyzing transcript...[/]")

    decisions = extractor.extract_decisions(transcript)
    action_items = extractor.extract_action_items(transcript)
    summary = summarizer.summarize(transcript, decisions, action_items)

    console.print()
    console.print(Panel(f"[bold]{summary.title}[/]", style="green"))
    console.print(summary.executive_summary)
    console.print()

    if summary.key_points:
        console.print("[bold]Key Points:[/]")
        for point in summary.key_points:
            console.print(f"  - {point}")

    if decisions:
        console.print()
        _display_decisions(decisions)

    if action_items:
        console.print()
        _display_actions(action_items)


def _display_result(result: MeetingResult) -> None:
    """Display processing results in the terminal."""
    s = result.summary
    console.print()
    console.print(Panel(f"[bold]{s.title}[/]", style="green"))
    console.print(f"{s.executive_summary}")
    console.print()
    console.print(
        f"[dim]{s.participant_count} speakers | {s.duration_minutes:.0f} min | "
        f"{len(result.transcript.segments)} segments[/]"
    )

    if s.key_points:
        console.print()
        console.print("[bold]Key Points:[/]")
        for point in s.key_points:
            console.print(f"  - {point}")

    if result.decisions:
        console.print()
        _display_decisions(result.decisions)

    if result.action_items:
        console.print()
        _display_actions(result.action_items)

    if result.qa_history:
        console.print()
        console.print("[bold]Q&A:[/]")
        for qa in result.qa_history:
            console.print(f"  [bold]Q:[/] {qa.question}")
            console.print(f"  [green]A:[/] {qa.answer}")
            console.print()


def _display_decisions(decisions) -> None:
    """Display decisions as a table."""
    table = Table(title="Decisions", show_header=True)
    table.add_column("#", width=3)
    table.add_column("Decision")
    table.add_column("Participants", style="dim")
    for d in decisions:
        table.add_row(str(d.id), d.text, ", ".join(d.participants) or "-")
    console.print(table)


def _display_actions(action_items) -> None:
    """Display action items as a table."""
    table = Table(title="Action Items", show_header=True)
    table.add_column("#", width=3)
    table.add_column("Task")
    table.add_column("Assignee")
    table.add_column("Deadline", style="dim")
    table.add_column("Priority")
    for a in action_items:
        table.add_row(
            str(a.id), a.task, a.assignee or "-",
            a.deadline or "-", a.priority.value,
        )
    console.print(table)


if __name__ == "__main__":
    cli()
