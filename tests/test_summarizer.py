"""Tests for meeting summarization."""

from unittest.mock import MagicMock

import pytest
from meeting_intelligence.models.meeting import (
    ActionItem,
    Decision,
    MeetingSummary,
)
from meeting_intelligence.understanding.summarizer import ChunkSummary, MeetingSummarizer


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def sample_summary():
    return MeetingSummary(
        title="Sprint Planning",
        executive_summary="Team discussed deployment.",
        key_points=["Deploy Friday"],
        topics_discussed=["Deployment"],
        participant_count=99,  # Will be overridden
        duration_minutes=999,  # Will be overridden
    )


class TestMeetingSummarizer:
    def test_single_pass_short_transcript(self, mock_llm, sample_transcript, sample_summary):
        mock_llm.count_tokens.return_value = 500  # Under 6000
        mock_llm.complete.return_value = sample_summary

        summarizer = MeetingSummarizer(mock_llm)
        result = summarizer.summarize(sample_transcript)

        assert result.title == "Sprint Planning"
        # Overridden with actual values
        assert result.participant_count == len(sample_transcript.speakers)
        assert result.duration_minutes == round(sample_transcript.duration_seconds / 60.0, 1)

    def test_map_reduce_long_transcript(self, mock_llm, sample_transcript, sample_summary):
        mock_llm.count_tokens.return_value = 7000  # Over 6000
        mock_llm.chunk_transcript.return_value = ["chunk1", "chunk2"]

        chunk1_summary = ChunkSummary(
            key_points=["Point A"], topics=["Topic A"], summary="Summary A"
        )
        chunk2_summary = ChunkSummary(
            key_points=["Point B"], topics=["Topic B"], summary="Summary B"
        )

        # Map calls return ChunkSummary, reduce call returns MeetingSummary
        mock_llm.complete.side_effect = [chunk1_summary, chunk2_summary, sample_summary]

        summarizer = MeetingSummarizer(mock_llm)
        result = summarizer.summarize(sample_transcript)

        assert result.title == "Sprint Planning"
        # 3 LLM calls: 2 map + 1 reduce
        assert mock_llm.complete.call_count == 3

    def test_overrides_participant_count(self, mock_llm, sample_transcript, sample_summary):
        mock_llm.count_tokens.return_value = 100
        sample_summary.participant_count = 999
        mock_llm.complete.return_value = sample_summary

        summarizer = MeetingSummarizer(mock_llm)
        result = summarizer.summarize(sample_transcript)

        # LLM's guess is overridden
        assert result.participant_count == 2  # sample_transcript has 2 speakers

    def test_overrides_duration(self, mock_llm, sample_transcript, sample_summary):
        mock_llm.count_tokens.return_value = 100
        sample_summary.duration_minutes = 999
        mock_llm.complete.return_value = sample_summary

        summarizer = MeetingSummarizer(mock_llm)
        result = summarizer.summarize(sample_transcript)

        expected = round(sample_transcript.duration_seconds / 60.0, 1)
        assert result.duration_minutes == expected

    def test_context_includes_decisions(self, mock_llm, sample_transcript, sample_summary):
        mock_llm.count_tokens.return_value = 100
        mock_llm.complete.return_value = sample_summary

        decisions = [Decision(id=1, text="Deploy Friday", confidence=0.9)]

        summarizer = MeetingSummarizer(mock_llm)
        summarizer.summarize(sample_transcript, decisions=decisions)

        user_prompt = mock_llm.complete.call_args[1]["user_prompt"]
        assert "Deploy Friday" in user_prompt

    def test_context_includes_action_items(self, mock_llm, sample_transcript, sample_summary):
        mock_llm.count_tokens.return_value = 100
        mock_llm.complete.return_value = sample_summary

        actions = [ActionItem(id=1, task="Send report", assignee="John")]

        summarizer = MeetingSummarizer(mock_llm)
        summarizer.summarize(sample_transcript, action_items=actions)

        user_prompt = mock_llm.complete.call_args[1]["user_prompt"]
        assert "Send report" in user_prompt
        assert "John" in user_prompt

    def test_no_context_when_none(self, mock_llm, sample_transcript, sample_summary):
        mock_llm.count_tokens.return_value = 100
        mock_llm.complete.return_value = sample_summary

        summarizer = MeetingSummarizer(mock_llm)
        summarizer.summarize(sample_transcript, decisions=None, action_items=None)

        user_prompt = mock_llm.complete.call_args[1]["user_prompt"]
        assert "Decisions made:" not in user_prompt
        assert "Action items:" not in user_prompt

    def test_build_context_empty_lists(self, mock_llm):
        summarizer = MeetingSummarizer(mock_llm)
        context = summarizer._build_context([], [])
        assert context == ""

    def test_build_context_none_assignee(self, mock_llm):
        summarizer = MeetingSummarizer(mock_llm)
        actions = [ActionItem(id=1, task="Do thing", assignee=None)]
        context = summarizer._build_context(None, actions)
        assert "unassigned" in context
