"""Tests for decision and action item extraction."""

from unittest.mock import MagicMock

import pytest
from meeting_intelligence.models.meeting import ActionItem, Decision, Priority
from meeting_intelligence.understanding.extractor import (
    ActionItemList,
    DecisionList,
    MeetingExtractor,
)


@pytest.fixture
def mock_llm():
    return MagicMock()


class TestMeetingExtractor:
    def test_extract_decisions(self, mock_llm, sample_transcript):
        decisions = [
            Decision(id=1, text="Deploy on Friday", confidence=0.9),
            Decision(id=2, text="Use Python", confidence=0.8),
        ]
        mock_llm.complete.return_value = DecisionList(decisions=decisions)
        mock_llm.count_tokens.return_value = 100

        extractor = MeetingExtractor(mock_llm)
        result = extractor.extract_decisions(sample_transcript)

        assert len(result) == 2
        assert result[0].text == "Deploy on Friday"
        assert result[1].text == "Use Python"

    def test_extract_decisions_empty(self, mock_llm, sample_transcript):
        mock_llm.complete.return_value = DecisionList(decisions=[])
        mock_llm.count_tokens.return_value = 100

        extractor = MeetingExtractor(mock_llm)
        result = extractor.extract_decisions(sample_transcript)

        assert result == []

    def test_extract_action_items(self, mock_llm, sample_transcript):
        items = [
            ActionItem(id=1, task="Send report", assignee="John", priority=Priority.HIGH),
            ActionItem(id=2, task="Update docs", assignee=None, priority=Priority.LOW),
        ]
        mock_llm.complete.return_value = ActionItemList(action_items=items)
        mock_llm.count_tokens.return_value = 100

        extractor = MeetingExtractor(mock_llm)
        result = extractor.extract_action_items(sample_transcript)

        assert len(result) == 2
        assert result[0].assignee == "John"
        assert result[1].assignee is None

    def test_extract_decisions_calls_llm_with_correct_model(self, mock_llm, sample_transcript):
        mock_llm.complete.return_value = DecisionList(decisions=[])
        mock_llm.count_tokens.return_value = 100

        extractor = MeetingExtractor(mock_llm)
        extractor.extract_decisions(sample_transcript)

        call_kwargs = mock_llm.complete.call_args[1]
        assert call_kwargs["response_model"] is DecisionList

    def test_extract_action_items_calls_llm_with_correct_model(self, mock_llm, sample_transcript):
        mock_llm.complete.return_value = ActionItemList(action_items=[])
        mock_llm.count_tokens.return_value = 100

        extractor = MeetingExtractor(mock_llm)
        extractor.extract_action_items(sample_transcript)

        call_kwargs = mock_llm.complete.call_args[1]
        assert call_kwargs["response_model"] is ActionItemList

    def test_long_transcript_uses_last_chunk(self, mock_llm, sample_transcript):
        mock_llm.count_tokens.return_value = 7000  # Over 6000 limit
        mock_llm.chunk_transcript.return_value = ["chunk1...", "chunk2 with recent content"]
        mock_llm.complete.return_value = DecisionList(decisions=[])

        extractor = MeetingExtractor(mock_llm)
        extractor.extract_decisions(sample_transcript)

        user_prompt = mock_llm.complete.call_args[1]["user_prompt"]
        assert "chunk2 with recent content" in user_prompt
        assert "[Earlier discussion omitted" in user_prompt

    def test_short_transcript_not_chunked(self, mock_llm, sample_transcript):
        mock_llm.count_tokens.return_value = 500
        mock_llm.complete.return_value = DecisionList(decisions=[])

        extractor = MeetingExtractor(mock_llm)
        extractor.extract_decisions(sample_transcript)

        mock_llm.chunk_transcript.assert_not_called()

    def test_transcript_text_includes_speakers(self, mock_llm, sample_transcript):
        mock_llm.count_tokens.return_value = 100
        mock_llm.complete.return_value = DecisionList(decisions=[])

        extractor = MeetingExtractor(mock_llm)
        extractor.extract_decisions(sample_transcript)

        user_prompt = mock_llm.complete.call_args[1]["user_prompt"]
        assert "SPEAKER_00" in user_prompt
        assert "SPEAKER_01" in user_prompt
