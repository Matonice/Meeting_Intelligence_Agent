"""JSON output writer for meeting results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..models.meeting import MeetingResult
from ..models.transcript import Transcript
from ..utils.logging import get_logger

logger = get_logger(__name__)


class JSONWriter:
    """Writes meeting results to JSON files."""

    def write(self, result: MeetingResult, output_dir: Path) -> Path:
        """Write complete meeting result to JSON.

        Creates: output_dir/meeting_YYYYMMDD_HHMMSS.json
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = result.processed_at.strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"meeting_{timestamp}.json"

        path.write_text(result.model_dump_json(indent=2))
        logger.info(f"JSON output: {path}")
        return path

    def write_transcript(self, transcript: Transcript, output_path: Path) -> Path:
        """Write just the transcript to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(transcript.model_dump_json(indent=2))
        logger.info(f"Transcript JSON: {output_path}")
        return output_path
