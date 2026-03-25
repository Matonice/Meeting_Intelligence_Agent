"""Text-to-Speech for meeting summary narration using edge-tts."""

from __future__ import annotations

import asyncio
from pathlib import Path

from ..config import TTSConfig
from ..models.meeting import MeetingSummary
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SummaryNarrator:
    """Converts meeting summaries to speech using edge-tts."""

    def __init__(self, config: TTSConfig) -> None:
        self.voice = config.voice
        self.rate = config.rate

    async def narrate(self, summary: MeetingSummary, output_path: Path) -> Path:
        """Generate audio narration of the meeting summary.

        Args:
            summary: The meeting summary to narrate.
            output_path: Where to save the audio file (.mp3).

        Returns:
            Path to the generated audio file.
        """
        import edge_tts

        text = self._format_summary_for_speech(summary)
        logger.info(f"Generating TTS narration ({len(text)} chars)...")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        communicate = edge_tts.Communicate(text=text, voice=self.voice, rate=self.rate)
        await communicate.save(str(output_path))

        logger.info(f"TTS output: {output_path}")
        return output_path

    def narrate_sync(self, summary: MeetingSummary, output_path: Path) -> Path:
        """Synchronous wrapper around async narrate."""
        return asyncio.run(self.narrate(summary, output_path))

    def _format_summary_for_speech(self, summary: MeetingSummary) -> str:
        """Convert structured summary into natural spoken text."""
        parts = [
            f"Meeting summary: {summary.title}.",
            "",
            summary.executive_summary,
            "",
            "Key points:",
        ]

        for i, point in enumerate(summary.key_points, 1):
            parts.append(f"Point {i}: {point}")

        if summary.topics_discussed:
            parts.append("")
            parts.append(f"Topics discussed included: {', '.join(summary.topics_discussed)}.")

        return "\n".join(parts)
