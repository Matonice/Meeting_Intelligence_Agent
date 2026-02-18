"""Tests for topic segmentation."""

from unittest.mock import MagicMock

import pytest

from meeting_intelligence.models.transcript import Transcript, TranscriptSegment, TopicBlock
from meeting_intelligence.postprocessing.segmenter import (
    TopicInfo,
    TopicList,
    TopicSegmenter,
)


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def multi_segment_transcript():
    """Transcript with 5 segments spanning 10 minutes."""
    segments = [
        TranscriptSegment(speaker="A", start=0.0, end=60.0, text="Intro discussion"),
        TranscriptSegment(speaker="B", start=65.0, end=180.0, text="Backend architecture"),
        TranscriptSegment(speaker="A", start=185.0, end=300.0, text="Database choices"),
        TranscriptSegment(speaker="B", start=310.0, end=450.0, text="Frontend framework"),
        TranscriptSegment(speaker="A", start=460.0, end=600.0, text="Deployment strategy"),
    ]
    return Transcript(
        segments=segments,
        speakers=["A", "B"],
        duration_seconds=600.0,
    )


class TestTopicSegmenter:
    def test_empty_transcript_returns_empty(self, mock_llm):
        transcript = Transcript(segments=[], speakers=[], duration_seconds=0.0)
        segmenter = TopicSegmenter(mock_llm)
        result = segmenter.segment(transcript)
        assert result == []

    def test_llm_segmentation(self, mock_llm, multi_segment_transcript):
        mock_llm.count_tokens.return_value = 500
        mock_llm.complete.return_value = TopicList(topics=[
            TopicInfo(topic="Introduction", summary="Opening remarks", start_index=0, end_index=0),
            TopicInfo(topic="Architecture", summary="Tech stack", start_index=1, end_index=2),
            TopicInfo(topic="Frontend & Deploy", summary="UI and ops", start_index=3, end_index=4),
        ])

        segmenter = TopicSegmenter(mock_llm)
        blocks = segmenter.segment(multi_segment_transcript)

        assert len(blocks) == 3
        assert blocks[0].topic == "Introduction"
        assert len(blocks[0].segments) == 1
        assert blocks[1].topic == "Architecture"
        assert len(blocks[1].segments) == 2
        assert blocks[2].topic == "Frontend & Deploy"
        assert len(blocks[2].segments) == 2

    def test_llm_segmentation_clamps_indices(self, mock_llm, sample_transcript):
        """Out-of-bound indices are clamped to valid range."""
        mock_llm.count_tokens.return_value = 100
        mock_llm.complete.return_value = TopicList(topics=[
            TopicInfo(topic="All", summary="Everything", start_index=-5, end_index=999),
        ])

        segmenter = TopicSegmenter(mock_llm)
        blocks = segmenter.segment(sample_transcript)

        assert len(blocks) == 1
        assert len(blocks[0].segments) == len(sample_transcript.segments)

    def test_llm_failure_falls_back_to_time_based(self, mock_llm, multi_segment_transcript):
        mock_llm.count_tokens.side_effect = Exception("API error")

        segmenter = TopicSegmenter(mock_llm)
        blocks = segmenter.segment(multi_segment_transcript)

        # Should get time-based blocks instead of crashing
        assert len(blocks) > 0
        assert all(isinstance(b, TopicBlock) for b in blocks)

    def test_time_based_fallback_single_block(self, mock_llm, sample_transcript):
        """Short transcript fits in one time block."""
        segmenter = TopicSegmenter(mock_llm)
        blocks = segmenter._time_based_fallback(sample_transcript, interval_minutes=5.0)

        # 10s transcript << 5min interval = single block
        assert len(blocks) == 1
        assert len(blocks[0].segments) == len(sample_transcript.segments)

    def test_time_based_fallback_multiple_blocks(self, mock_llm, multi_segment_transcript):
        """10-minute transcript with 3-minute intervals."""
        segmenter = TopicSegmenter(mock_llm)
        blocks = segmenter._time_based_fallback(multi_segment_transcript, interval_minutes=3.0)

        # 600s / 180s intervals = should produce multiple blocks
        assert len(blocks) >= 2
        total_segs = sum(len(b.segments) for b in blocks)
        assert total_segs == len(multi_segment_transcript.segments)

    def test_time_based_fallback_empty_transcript(self, mock_llm):
        transcript = Transcript(segments=[], speakers=[], duration_seconds=0.0)
        segmenter = TopicSegmenter(mock_llm)
        blocks = segmenter._time_based_fallback(transcript)
        assert blocks == []

    def test_topic_block_start_end_from_segments(self, mock_llm, multi_segment_transcript):
        mock_llm.count_tokens.return_value = 100
        mock_llm.complete.return_value = TopicList(topics=[
            TopicInfo(topic="First", summary="s", start_index=0, end_index=1),
        ])

        segmenter = TopicSegmenter(mock_llm)
        blocks = segmenter.segment(multi_segment_transcript)

        assert blocks[0].start == multi_segment_transcript.segments[0].start
        assert blocks[0].end == multi_segment_transcript.segments[1].end

    def test_long_transcript_chunked(self, mock_llm, multi_segment_transcript):
        mock_llm.count_tokens.return_value = 7000
        mock_llm.chunk_transcript.return_value = ["chunk1", "chunk2"]
        mock_llm.complete.return_value = TopicList(topics=[
            TopicInfo(topic="Overview", summary="s", start_index=0, end_index=4),
        ])

        segmenter = TopicSegmenter(mock_llm)
        segmenter.segment(multi_segment_transcript)

        # Uses first chunk (unlike extractor which uses last)
        user_prompt = mock_llm.complete.call_args[1]["user_prompt"]
        assert "chunk1" in user_prompt
