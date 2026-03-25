"""Topic segmentation of meeting transcripts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..models.transcript import TopicBlock, Transcript
from ..understanding.llm_client import LLMClient
from ..utils.logging import get_logger

logger = get_logger(__name__)


SEGMENTATION_SYSTEM_PROMPT = """You are a meeting analyst. Given a numbered meeting transcript, 
identify distinct topics discussed.

For each topic, provide:
- topic: Short topic name (2-5 words)
- summary: 1-2 sentence summary of what was discussed
- start_index: Index of the first segment in this topic
- end_index: Index of the last segment in this topic (inclusive)

Topics should be non-overlapping and cover the entire transcript.
Return valid JSON."""


class TopicInfo(BaseModel):
    topic: str
    summary: str
    start_index: int
    end_index: int


class TopicList(BaseModel):
    topics: list[TopicInfo] = Field(default_factory=list)


class TopicSegmenter:
    """Segments transcript into topic blocks using LLM analysis."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def segment(self, transcript: Transcript) -> list[TopicBlock]:
        """Identify topic boundaries and label topics.

        Falls back to time-based segmentation if LLM is unavailable.
        """
        if not transcript.segments:
            return []

        try:
            return self._llm_segment(transcript)
        except Exception as e:
            logger.warning(f"LLM segmentation failed: {e}. Using time-based fallback.")
            return self._time_based_fallback(transcript)

    def _llm_segment(self, transcript: Transcript) -> list[TopicBlock]:
        """Use GPT-4 to identify topic boundaries."""
        numbered_lines = []
        for i, seg in enumerate(transcript.segments):
            numbered_lines.append(f"[{i}] {seg.speaker}: {seg.text}")
        text = "\n".join(numbered_lines)

        # Chunk if needed
        token_count = self.llm.count_tokens(text)
        if token_count > 6000:
            chunks = self.llm.chunk_transcript(text, max_tokens=5500)
            text = chunks[0]  # Use first chunk for topic overview
            if len(chunks) > 1:
                text += "\n...[transcript continues]"

        result = self.llm.complete(
            system_prompt=SEGMENTATION_SYSTEM_PROMPT,
            user_prompt=text,
            response_model=TopicList,
        )

        # Convert to TopicBlock objects
        blocks = []
        for info in result.topics:
            start_idx = max(0, min(info.start_index, len(transcript.segments) - 1))
            end_idx = max(start_idx, min(info.end_index, len(transcript.segments) - 1))
            segs = transcript.segments[start_idx : end_idx + 1]
            if segs:
                blocks.append(
                    TopicBlock(
                        topic=info.topic,
                        summary=info.summary,
                        segments=segs,
                        start=segs[0].start,
                        end=segs[-1].end,
                    )
                )

        logger.info(f"Topic segmentation: {len(blocks)} topics identified")
        return blocks

    def _time_based_fallback(
        self, transcript: Transcript, interval_minutes: float = 5.0
    ) -> list[TopicBlock]:
        """Simple time-based segmentation as fallback."""
        interval_sec = interval_minutes * 60
        blocks: list[TopicBlock] = []
        current_segments = []
        block_start = 0.0

        for seg in transcript.segments:
            if seg.start - block_start >= interval_sec and current_segments:
                blocks.append(
                    TopicBlock(
                        topic=f"Discussion ({block_start / 60:.0f}-{seg.start / 60:.0f} min)",
                        summary="",
                        segments=current_segments,
                        start=current_segments[0].start,
                        end=current_segments[-1].end,
                    )
                )
                current_segments = []
                block_start = seg.start
            current_segments.append(seg)

        if current_segments:
            blocks.append(
                TopicBlock(
                    topic=(
                        f"Discussion ({block_start / 60:.0f}-"
                        f"{current_segments[-1].end / 60:.0f} min)"
                    ),
                    summary="",
                    segments=current_segments,
                    start=current_segments[0].start,
                    end=current_segments[-1].end,
                )
            )

        return blocks
