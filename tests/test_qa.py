"""Tests for Q&A over meeting transcripts."""

from unittest.mock import MagicMock

import pytest

from meeting_intelligence.models.meeting import QAResponse
from meeting_intelligence.models.transcript import Transcript, TranscriptSegment
from meeting_intelligence.understanding.qa import MeetingQA


@pytest.fixture
def mock_llm():
    return MagicMock()


class TestMeetingQA:
    def test_ask_returns_response(self, mock_llm, sample_transcript):
        mock_llm.count_tokens.return_value = 100
        mock_llm.complete.return_value = QAResponse(
            question="What about deployment?",
            answer="The team decided to deploy on Friday.",
            confidence=0.9,
        )

        qa = MeetingQA(mock_llm)
        result = qa.ask("What about deployment?", sample_transcript)

        assert result.answer == "The team decided to deploy on Friday."
        assert result.confidence == 0.9

    def test_ask_overwrites_question_field(self, mock_llm, sample_transcript):
        """Ensures returned question matches what was asked, even if LLM changes it."""
        mock_llm.count_tokens.return_value = 100
        mock_llm.complete.return_value = QAResponse(
            question="LLM changed the question",
            answer="Answer here.",
            confidence=0.8,
        )

        qa = MeetingQA(mock_llm)
        result = qa.ask("My original question?", sample_transcript)

        assert result.question == "My original question?"

    def test_ask_with_preloaded_transcript(self, mock_llm, sample_transcript):
        mock_llm.count_tokens.return_value = 100
        mock_llm.complete.return_value = QAResponse(
            question="Q", answer="A", confidence=0.5,
        )

        qa = MeetingQA(mock_llm)
        qa.load_transcript(sample_transcript)
        result = qa.ask("Q")

        assert result.answer == "A"

    def test_ask_no_transcript_raises(self, mock_llm):
        qa = MeetingQA(mock_llm)
        with pytest.raises(ValueError, match="No transcript loaded"):
            qa.ask("What happened?")

    def test_ask_argument_overrides_preloaded(self, mock_llm, sample_transcript):
        mock_llm.count_tokens.return_value = 100
        mock_llm.complete.return_value = QAResponse(
            question="Q", answer="From arg transcript", confidence=0.7,
        )

        other_transcript = Transcript(
            segments=[
                TranscriptSegment(speaker="X", start=0, end=1, text="Other content"),
            ],
            speakers=["X"],
            duration_seconds=1.0,
        )

        qa = MeetingQA(mock_llm)
        qa.load_transcript(sample_transcript)
        result = qa.ask("Q", transcript=other_transcript)

        # Should use the other_transcript
        user_prompt = mock_llm.complete.call_args[1]["user_prompt"]
        assert "Other content" in user_prompt

    def test_ask_includes_numbered_segments(self, mock_llm, sample_transcript):
        mock_llm.count_tokens.return_value = 100
        mock_llm.complete.return_value = QAResponse(
            question="Q", answer="A", confidence=0.5,
        )

        qa = MeetingQA(mock_llm)
        qa.ask("Q", sample_transcript)

        user_prompt = mock_llm.complete.call_args[1]["user_prompt"]
        assert "[0]" in user_prompt
        assert "[1]" in user_prompt

    def test_ask_long_transcript_uses_last_chunk(self, mock_llm, sample_transcript):
        mock_llm.count_tokens.return_value = 7000
        mock_llm.chunk_transcript.return_value = ["old stuff", "recent content"]
        mock_llm.complete.return_value = QAResponse(
            question="Q", answer="A", confidence=0.5,
        )

        qa = MeetingQA(mock_llm)
        qa.ask("Q", sample_transcript)

        user_prompt = mock_llm.complete.call_args[1]["user_prompt"]
        assert "recent content" in user_prompt
        assert "[Earlier discussion omitted]" in user_prompt

    def test_load_transcript_can_be_overwritten(self, mock_llm):
        t1 = Transcript(
            segments=[TranscriptSegment(speaker="A", start=0, end=1, text="First")],
            speakers=["A"], duration_seconds=1.0,
        )
        t2 = Transcript(
            segments=[TranscriptSegment(speaker="B", start=0, end=1, text="Second")],
            speakers=["B"], duration_seconds=1.0,
        )

        mock_llm.count_tokens.return_value = 100
        mock_llm.complete.return_value = QAResponse(
            question="Q", answer="A", confidence=0.5,
        )

        qa = MeetingQA(mock_llm)
        qa.load_transcript(t1)
        qa.load_transcript(t2)
        qa.ask("Q")

        user_prompt = mock_llm.complete.call_args[1]["user_prompt"]
        assert "Second" in user_prompt
        assert "First" not in user_prompt
