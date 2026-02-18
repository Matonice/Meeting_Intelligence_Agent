"""Transcript data models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Word(BaseModel):
    """A single word with timing information."""

    text: str
    start: float
    end: float
    confidence: float = 0.0


class DiarizationSegment(BaseModel):
    """A speaker turn from diarization."""

    speaker: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    """A speaker-attributed transcript segment."""

    speaker: str
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)
    confidence: float = 0.0


class TopicBlock(BaseModel):
    """A block of transcript segments grouped by topic."""

    topic: str
    summary: str
    segments: list[TranscriptSegment]
    start: float
    end: float


class Transcript(BaseModel):
    """The full assembled transcript."""

    segments: list[TranscriptSegment]
    speakers: list[str]
    language: str = "en"
    duration_seconds: float
    topic_blocks: list[TopicBlock] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        return " ".join(seg.text for seg in self.segments)

    @property
    def speaker_text(self) -> str:
        lines = []
        for seg in self.segments:
            lines.append(f"[{seg.start:.1f}s] {seg.speaker}: {seg.text}")
        return "\n".join(lines)
