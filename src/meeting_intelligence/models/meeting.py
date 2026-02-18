"""Meeting understanding data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from .transcript import Transcript


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Decision(BaseModel):
    """A decision extracted from the meeting."""

    id: int
    text: str
    context: str = ""
    timestamp: float | None = None
    participants: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ActionItem(BaseModel):
    """An action item / task extracted from the meeting."""

    id: int
    task: str
    assignee: str | None = None
    deadline: str | None = None
    priority: Priority = Priority.MEDIUM
    context: str = ""
    timestamp: float | None = None
    status: str = "pending"
    confidence: float = 0.0


class MeetingSummary(BaseModel):
    """Structured summary of the meeting."""

    title: str
    executive_summary: str
    key_points: list[str]
    topics_discussed: list[str]
    participant_count: int
    duration_minutes: float


class QAResponse(BaseModel):
    """Response to a question about the meeting."""

    question: str
    answer: str
    relevant_segments: list[int] = Field(default_factory=list)
    confidence: float = 0.0


class MeetingResult(BaseModel):
    """The complete output of the meeting intelligence pipeline."""

    audio_file: str
    processed_at: datetime = Field(default_factory=datetime.now)
    transcript: Transcript
    summary: MeetingSummary
    decisions: list[Decision]
    action_items: list[ActionItem]
    qa_history: list[QAResponse] = Field(default_factory=list)
