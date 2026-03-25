"""Tests for Pydantic data models."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from meeting_intelligence.models.audio import AudioData, AudioMetadata, SpeechSegment
from meeting_intelligence.models.meeting import (
    ActionItem,
    Decision,
    MeetingResult,
    MeetingSummary,
    Priority,
    QAResponse,
)
from meeting_intelligence.models.transcript import (
    DiarizationSegment,
    Transcript,
    TranscriptSegment,
    Word,
)


class TestSpeechSegment:
    def test_duration_property(self):
        seg = SpeechSegment(start=1.0, end=3.5)
        assert seg.duration == 2.5

    def test_duration_zero_length(self):
        seg = SpeechSegment(start=5.0, end=5.0)
        assert seg.duration == 0.0

    def test_default_confidence(self):
        seg = SpeechSegment(start=0.0, end=1.0)
        assert seg.confidence == 1.0


class TestAudioMetadata:
    def test_creation(self):
        meta = AudioMetadata(
            file_path=Path("/tmp/test.wav"),
            duration_seconds=10.5,
            sample_rate=16000,
            channels=2,
            format="wav",
            file_size_bytes=336000,
        )
        assert meta.duration_seconds == 10.5
        assert meta.channels == 2

    def test_serialization(self):
        meta = AudioMetadata(
            file_path=Path("/tmp/test.wav"),
            duration_seconds=5.0,
            sample_rate=16000,
            channels=1,
            format="wav",
            file_size_bytes=160000,
        )
        data = json.loads(meta.model_dump_json())
        assert data["sample_rate"] == 16000


class TestAudioData:
    def test_samples_excluded_from_serialization(self):
        samples = np.zeros(16000, dtype=np.float32)
        audio = AudioData(
            samples=samples,
            sample_rate=16000,
            duration_seconds=1.0,
            metadata=AudioMetadata(
                file_path=Path("/tmp/t.wav"),
                duration_seconds=1.0,
                sample_rate=16000,
                channels=1,
                format="wav",
                file_size_bytes=32000,
            ),
        )
        data = audio.model_dump()
        assert "samples" not in data

    def test_numpy_array_accepted(self):
        samples = np.random.randn(8000).astype(np.float32)
        audio = AudioData(
            samples=samples,
            sample_rate=16000,
            duration_seconds=0.5,
            metadata=AudioMetadata(
                file_path=Path("/tmp/t.wav"),
                duration_seconds=0.5,
                sample_rate=16000,
                channels=1,
                format="wav",
                file_size_bytes=16000,
            ),
        )
        assert len(audio.samples) == 8000


class TestWord:
    def test_default_confidence(self):
        w = Word(text="hello", start=0.0, end=0.5)
        assert w.confidence == 0.0

    def test_creation(self):
        w = Word(text="world", start=1.0, end=1.5, confidence=0.95)
        assert w.text == "world"
        assert w.confidence == 0.95


class TestTranscript:
    def test_full_text_with_segments(self):
        t = Transcript(
            segments=[
                TranscriptSegment(speaker="A", start=0, end=1, text="Hello"),
                TranscriptSegment(speaker="B", start=1, end=2, text="World"),
            ],
            speakers=["A", "B"],
            duration_seconds=2.0,
        )
        assert t.full_text == "Hello World"

    def test_full_text_empty_segments(self):
        t = Transcript(segments=[], speakers=[], duration_seconds=0.0)
        assert t.full_text == ""

    def test_speaker_text_format(self):
        t = Transcript(
            segments=[
                TranscriptSegment(speaker="SPEAKER_00", start=1.5, end=3.0, text="Hi there"),
            ],
            speakers=["SPEAKER_00"],
            duration_seconds=3.0,
        )
        assert t.speaker_text == "[1.5s] SPEAKER_00: Hi there"

    def test_speaker_text_empty(self):
        t = Transcript(segments=[], speakers=[], duration_seconds=0.0)
        assert t.speaker_text == ""

    def test_default_topic_blocks(self):
        t = Transcript(segments=[], speakers=[], duration_seconds=0.0)
        assert t.topic_blocks == []

    def test_default_language(self):
        t = Transcript(segments=[], speakers=[], duration_seconds=0.0)
        assert t.language == "en"

    def test_serialization_roundtrip(self):
        t = Transcript(
            segments=[
                TranscriptSegment(
                    speaker="A",
                    start=0.0,
                    end=1.0,
                    text="Test",
                    words=[Word(text="Test", start=0.0, end=1.0, confidence=0.9)],
                ),
            ],
            speakers=["A"],
            duration_seconds=1.0,
        )
        json_str = t.model_dump_json()
        restored = Transcript.model_validate_json(json_str)
        assert restored.segments[0].text == "Test"
        assert restored.segments[0].words[0].confidence == 0.9


class TestDiarizationSegment:
    def test_creation(self):
        seg = DiarizationSegment(speaker="SPEAKER_01", start=2.0, end=5.0)
        assert seg.speaker == "SPEAKER_01"
        assert seg.end - seg.start == 3.0


class TestPriority:
    def test_enum_values(self):
        assert Priority.HIGH == "high"
        assert Priority.MEDIUM == "medium"
        assert Priority.LOW == "low"

    def test_string_comparison(self):
        assert Priority.HIGH.value == "high"


class TestDecision:
    def test_defaults(self):
        d = Decision(id=1, text="Deploy Friday")
        assert d.context == ""
        assert d.timestamp is None
        assert d.participants == []
        assert d.confidence == 0.0

    def test_with_all_fields(self):
        d = Decision(
            id=1,
            text="Deploy",
            context="Sprint meeting",
            timestamp=45.2,
            participants=["Alice", "Bob"],
            confidence=0.9,
        )
        assert len(d.participants) == 2


class TestActionItem:
    def test_defaults(self):
        a = ActionItem(id=1, task="Send report")
        assert a.assignee is None
        assert a.deadline is None
        assert a.priority == Priority.MEDIUM
        assert a.status == "pending"

    def test_with_all_fields(self):
        a = ActionItem(
            id=1,
            task="Send report",
            assignee="John",
            deadline="Friday",
            priority=Priority.HIGH,
        )
        assert a.assignee == "John"
        assert a.priority == Priority.HIGH


class TestMeetingSummary:
    def test_creation(self):
        s = MeetingSummary(
            title="Sprint Planning",
            executive_summary="Discussed deployment.",
            key_points=["Deploy Friday"],
            topics_discussed=["Deployment"],
            participant_count=3,
            duration_minutes=30.0,
        )
        assert s.participant_count == 3


class TestQAResponse:
    def test_defaults(self):
        qa = QAResponse(question="What?", answer="Nothing.")
        assert qa.relevant_segments == []
        assert qa.confidence == 0.0


class TestMeetingResult:
    def test_processed_at_auto_set(self):
        t = Transcript(segments=[], speakers=[], duration_seconds=0.0)
        s = MeetingSummary(
            title="T",
            executive_summary="S",
            key_points=[],
            topics_discussed=[],
            participant_count=0,
            duration_minutes=0,
        )
        r = MeetingResult(
            audio_file="test.wav",
            transcript=t,
            summary=s,
            decisions=[],
            action_items=[],
        )
        assert isinstance(r.processed_at, datetime)

    def test_qa_history_default_empty(self):
        t = Transcript(segments=[], speakers=[], duration_seconds=0.0)
        s = MeetingSummary(
            title="T",
            executive_summary="S",
            key_points=[],
            topics_discussed=[],
            participant_count=0,
            duration_minutes=0,
        )
        r = MeetingResult(
            audio_file="test.wav",
            transcript=t,
            summary=s,
            decisions=[],
            action_items=[],
        )
        assert r.qa_history == []

    def test_full_serialization_roundtrip(self, sample_meeting_result):
        json_str = sample_meeting_result.model_dump_json()
        restored = MeetingResult.model_validate_json(json_str)
        assert restored.summary.title == sample_meeting_result.summary.title
        assert len(restored.decisions) == len(sample_meeting_result.decisions)
        assert len(restored.action_items) == len(sample_meeting_result.action_items)
