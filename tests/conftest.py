"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from meeting_intelligence.config import AppConfig, load_config
from meeting_intelligence.models.audio import AudioData, AudioMetadata
from meeting_intelligence.models.meeting import (
    ActionItem,
    Decision,
    MeetingResult,
    MeetingSummary,
    Priority,
)
from meeting_intelligence.models.transcript import (
    DiarizationSegment,
    Transcript,
    TranscriptSegment,
    Word,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def default_config() -> AppConfig:
    return AppConfig()


@pytest.fixture
def sample_audio() -> AudioData:
    """Generate a 3-second silent audio sample."""
    sample_rate = 16000
    duration = 3.0
    samples = np.zeros(int(sample_rate * duration), dtype=np.float32)
    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        duration_seconds=duration,
        metadata=AudioMetadata(
            file_path=Path("/tmp/test.wav"),
            duration_seconds=duration,
            sample_rate=sample_rate,
            channels=1,
            format="wav",
            file_size_bytes=96000,
        ),
    )


@pytest.fixture
def sample_diarization_segments() -> list[DiarizationSegment]:
    return [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=3.5),
        DiarizationSegment(speaker="SPEAKER_01", start=3.8, end=7.0),
        DiarizationSegment(speaker="SPEAKER_00", start=7.2, end=10.0),
    ]


@pytest.fixture
def sample_asr_segments() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            speaker="",
            start=0.0,
            end=3.2,
            text="Hello everyone let's get started",
            words=[
                Word(text="Hello", start=0.0, end=0.4, confidence=0.95),
                Word(text="everyone", start=0.5, end=1.0, confidence=0.92),
                Word(text="let's", start=1.1, end=1.4, confidence=0.90),
                Word(text="get", start=1.5, end=1.7, confidence=0.93),
                Word(text="started", start=1.8, end=2.3, confidence=0.91),
            ],
        ),
        TranscriptSegment(
            speaker="",
            start=3.9,
            end=6.5,
            text="Sure we should deploy on Friday",
            words=[
                Word(text="Sure", start=3.9, end=4.2, confidence=0.88),
                Word(text="we", start=4.3, end=4.4, confidence=0.90),
                Word(text="should", start=4.5, end=4.8, confidence=0.87),
                Word(text="deploy", start=4.9, end=5.3, confidence=0.92),
                Word(text="on", start=5.4, end=5.5, confidence=0.95),
                Word(text="Friday", start=5.6, end=6.1, confidence=0.94),
            ],
        ),
        TranscriptSegment(
            speaker="",
            start=7.5,
            end=9.8,
            text="Agreed John please send the report",
            words=[
                Word(text="Agreed", start=7.5, end=7.9, confidence=0.90),
                Word(text="John", start=8.0, end=8.3, confidence=0.85),
                Word(text="please", start=8.4, end=8.7, confidence=0.92),
                Word(text="send", start=8.8, end=9.1, confidence=0.88),
                Word(text="the", start=9.2, end=9.3, confidence=0.95),
                Word(text="report", start=9.4, end=9.8, confidence=0.91),
            ],
        ),
    ]


@pytest.fixture
def sample_transcript(sample_asr_segments) -> Transcript:
    """A pre-assembled transcript with speaker labels."""
    segments = [
        sample_asr_segments[0].model_copy(update={"speaker": "SPEAKER_00"}),
        sample_asr_segments[1].model_copy(update={"speaker": "SPEAKER_01"}),
        sample_asr_segments[2].model_copy(update={"speaker": "SPEAKER_00"}),
    ]
    return Transcript(
        segments=segments,
        speakers=["SPEAKER_00", "SPEAKER_01"],
        language="en",
        duration_seconds=10.0,
    )


@pytest.fixture
def sample_meeting_result(sample_transcript) -> MeetingResult:
    return MeetingResult(
        audio_file="/tmp/test_meeting.wav",
        transcript=sample_transcript,
        summary=MeetingSummary(
            title="Sprint Planning Meeting",
            executive_summary="Team discussed deployment timeline and assigned tasks.",
            key_points=["Deploy on Friday", "John to send report"],
            topics_discussed=["Deployment", "Task assignment"],
            participant_count=2,
            duration_minutes=0.2,
        ),
        decisions=[
            Decision(
                id=1,
                text="Deploy on Friday",
                participants=["SPEAKER_00", "SPEAKER_01"],
                confidence=0.9,
            ),
        ],
        action_items=[
            ActionItem(
                id=1,
                task="Send the report",
                assignee="John",
                priority=Priority.HIGH,
                confidence=0.85,
            ),
        ],
    )
