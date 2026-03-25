"""Tests for text-to-speech summary narration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from meeting_intelligence.config import TTSConfig
from meeting_intelligence.models.meeting import MeetingSummary
from meeting_intelligence.tts.speaker import SummaryNarrator


@pytest.fixture
def tts_config():
    return TTSConfig(voice="en-US-TestVoice", rate="+10%")


@pytest.fixture
def sample_summary():
    return MeetingSummary(
        title="Sprint Planning",
        executive_summary="Team discussed deployment timeline.",
        key_points=["Deploy Friday", "Update docs"],
        topics_discussed=["Deployment", "Documentation"],
        participant_count=3,
        duration_minutes=30.0,
    )


class TestSummaryNarrator:
    def test_format_summary_includes_title(self, tts_config, sample_summary):
        narrator = SummaryNarrator(tts_config)
        text = narrator._format_summary_for_speech(sample_summary)

        assert "Sprint Planning" in text

    def test_format_summary_includes_executive_summary(self, tts_config, sample_summary):
        narrator = SummaryNarrator(tts_config)
        text = narrator._format_summary_for_speech(sample_summary)

        assert "Team discussed deployment timeline." in text

    def test_format_summary_includes_key_points(self, tts_config, sample_summary):
        narrator = SummaryNarrator(tts_config)
        text = narrator._format_summary_for_speech(sample_summary)

        assert "Point 1: Deploy Friday" in text
        assert "Point 2: Update docs" in text

    def test_format_summary_includes_topics(self, tts_config, sample_summary):
        narrator = SummaryNarrator(tts_config)
        text = narrator._format_summary_for_speech(sample_summary)

        assert "Deployment" in text
        assert "Documentation" in text

    def test_format_summary_empty_topics_skipped(self, tts_config):
        summary = MeetingSummary(
            title="Test",
            executive_summary="Test.",
            key_points=[],
            topics_discussed=[],
            participant_count=1,
            duration_minutes=1.0,
        )
        narrator = SummaryNarrator(tts_config)
        text = narrator._format_summary_for_speech(summary)

        assert "Topics discussed" not in text

    def test_format_summary_empty_key_points(self, tts_config):
        summary = MeetingSummary(
            title="Test",
            executive_summary="Test.",
            key_points=[],
            topics_discussed=["Topic"],
            participant_count=1,
            duration_minutes=1.0,
        )
        narrator = SummaryNarrator(tts_config)
        text = narrator._format_summary_for_speech(summary)

        assert "Key points:" in text
        assert "Point 1" not in text

    def test_narrate_sync_calls_edge_tts(self, tts_config, sample_summary, tmp_path):
        mock_communicate = AsyncMock()
        mock_edge_tts = MagicMock()
        mock_edge_tts.Communicate.return_value = mock_communicate

        narrator = SummaryNarrator(tts_config)
        output = tmp_path / "summary.mp3"

        # Patch the import inside the narrate method
        with patch.dict("sys.modules", {"edge_tts": mock_edge_tts}):
            narrator.narrate_sync(sample_summary, output)

        mock_edge_tts.Communicate.assert_called_once()
        call_kwargs = mock_edge_tts.Communicate.call_args[1]
        assert call_kwargs["voice"] == "en-US-TestVoice"
        assert call_kwargs["rate"] == "+10%"
        mock_communicate.save.assert_called_once_with(str(output))

    def test_narrate_creates_parent_dirs(self, tts_config, sample_summary, tmp_path):
        mock_communicate = AsyncMock()
        mock_edge_tts = MagicMock()
        mock_edge_tts.Communicate.return_value = mock_communicate

        narrator = SummaryNarrator(tts_config)
        output = tmp_path / "nested" / "dir" / "summary.mp3"

        with patch.dict("sys.modules", {"edge_tts": mock_edge_tts}):
            narrator.narrate_sync(sample_summary, output)

        assert output.parent.exists()

    def test_narrate_returns_path(self, tts_config, sample_summary, tmp_path):
        mock_communicate = AsyncMock()
        mock_edge_tts = MagicMock()
        mock_edge_tts.Communicate.return_value = mock_communicate

        narrator = SummaryNarrator(tts_config)
        output = tmp_path / "summary.mp3"

        with patch.dict("sys.modules", {"edge_tts": mock_edge_tts}):
            result = narrator.narrate_sync(sample_summary, output)

        assert result == output
