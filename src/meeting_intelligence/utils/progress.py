"""Progress reporting for pipeline stages."""

from __future__ import annotations

from typing import Protocol

from rich.console import Console


class ProgressCallback(Protocol):
    """Protocol for pipeline progress reporting."""

    def on_stage_start(self, stage: str, description: str) -> None: ...
    def on_stage_complete(self, stage: str, duration_seconds: float) -> None: ...
    def on_stage_error(self, stage: str, error: Exception) -> None: ...


class RichProgressCallback:
    """Rich-based progress reporting for CLI usage."""

    def __init__(self) -> None:
        self.console = Console()

    def on_stage_start(self, stage: str, description: str) -> None:
        self.console.print(f"[bold blue]▶[/] {description}")

    def on_stage_complete(self, stage: str, duration_seconds: float) -> None:
        self.console.print(f"  [green]✓[/] {stage} [{duration_seconds:.1f}s]")

    def on_stage_error(self, stage: str, error: Exception) -> None:
        self.console.print(f"  [red]✗[/] {stage}: {error}")
