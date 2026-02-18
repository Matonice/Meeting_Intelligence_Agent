"""Markdown report writer for meeting results."""

from __future__ import annotations

from pathlib import Path

from ..models.meeting import MeetingResult
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MarkdownWriter:
    """Writes meeting results as Markdown reports."""

    def write(self, result: MeetingResult, output_dir: Path) -> Path:
        """Write meeting report as Markdown.

        Creates: output_dir/meeting_YYYYMMDD_HHMMSS.md
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = result.processed_at.strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"meeting_{timestamp}.md"

        content = self._render(result)
        path.write_text(content)
        logger.info(f"Markdown output: {path}")
        return path

    def _render(self, result: MeetingResult) -> str:
        """Render MeetingResult to Markdown string."""
        s = result.summary
        lines = [
            f"# {s.title}",
            "",
            f"**Date**: {result.processed_at.strftime('%Y-%m-%d %H:%M')}  ",
            f"**Duration**: {s.duration_minutes:.0f} min  ",
            f"**Speakers**: {s.participant_count}  ",
            f"**Source**: `{result.audio_file}`",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            s.executive_summary,
            "",
            "## Key Points",
            "",
        ]

        for point in s.key_points:
            lines.append(f"- {point}")

        lines.extend(["", "## Topics Discussed", ""])
        for topic in s.topics_discussed:
            lines.append(f"- {topic}")

        # Decisions table
        if result.decisions:
            lines.extend([
                "",
                "## Decisions",
                "",
                "| # | Decision | Participants |",
                "|---|----------|-------------|",
            ])
            for d in result.decisions:
                participants = ", ".join(d.participants) if d.participants else "-"
                lines.append(f"| {d.id} | {d.text} | {participants} |")

        # Action items table
        if result.action_items:
            lines.extend([
                "",
                "## Action Items",
                "",
                "| # | Task | Assignee | Deadline | Priority |",
                "|---|------|----------|----------|----------|",
            ])
            for a in result.action_items:
                lines.append(
                    f"| {a.id} | {a.task} | {a.assignee or '-'} | "
                    f"{a.deadline or '-'} | {a.priority.value} |"
                )

        # Full transcript
        lines.extend(["", "## Full Transcript", ""])
        for seg in result.transcript.segments:
            timestamp = f"{seg.start:.1f}s"
            lines.append(f"**[{timestamp}] {seg.speaker}**: {seg.text}  ")

        lines.append("")
        return "\n".join(lines)
