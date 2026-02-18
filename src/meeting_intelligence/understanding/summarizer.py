"""Meeting summary generation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..models.meeting import ActionItem, Decision, MeetingSummary
from ..models.transcript import Transcript
from ..utils.logging import get_logger
from .llm_client import LLMClient

logger = get_logger(__name__)


SUMMARY_SYSTEM_PROMPT = """You are a meeting analyst. Generate a structured summary of this meeting.

Provide:
- title: A short descriptive title for the meeting (max 10 words)
- executive_summary: 2-3 sentence overview of what was discussed and decided
- key_points: List of 3-7 most important points from the meeting
- topics_discussed: List of main topics/themes covered
- participant_count: Number of unique speakers
- duration_minutes: Meeting duration in minutes

Be concise and factual. Base your summary only on the transcript content. Return valid JSON."""


class ChunkSummary(BaseModel):
    """Summary of a transcript chunk for map-reduce."""

    key_points: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    summary: str = ""


class MeetingSummarizer:
    """Generates structured meeting summaries using GPT-4."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def summarize(
        self,
        transcript: Transcript,
        decisions: list[Decision] | None = None,
        action_items: list[ActionItem] | None = None,
    ) -> MeetingSummary:
        """Generate a structured meeting summary.

        Uses single-pass for short meetings, map-reduce for long ones.
        """
        logger.info("Generating meeting summary...")
        transcript_text = transcript.speaker_text
        token_count = self.llm.count_tokens(transcript_text)

        # Build context from decisions/actions
        context = self._build_context(decisions, action_items)

        if token_count <= 6000:
            return self._single_pass_summary(transcript_text, context, transcript)
        else:
            return self._map_reduce_summary(transcript_text, context, transcript)

    def _single_pass_summary(
        self, transcript_text: str, context: str, transcript: Transcript
    ) -> MeetingSummary:
        """Summarize when transcript fits in token window."""
        user_prompt = f"Meeting transcript:\n\n{transcript_text}"
        if context:
            user_prompt += f"\n\nAdditional context:\n{context}"

        result = self.llm.complete(
            system_prompt=SUMMARY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=MeetingSummary,
        )
        # Override with accurate metadata
        result.participant_count = len(transcript.speakers)
        result.duration_minutes = round(transcript.duration_seconds / 60.0, 1)
        return result

    def _map_reduce_summary(
        self, transcript_text: str, context: str, transcript: Transcript
    ) -> MeetingSummary:
        """Summarize long transcripts via map-reduce."""
        chunks = self.llm.chunk_transcript(transcript_text, max_tokens=5000)
        logger.info(f"Map-reduce: {len(chunks)} chunks")

        # Map phase: summarize each chunk
        chunk_summaries: list[ChunkSummary] = []
        for i, chunk in enumerate(chunks):
            logger.debug(f"Summarizing chunk {i + 1}/{len(chunks)}")
            cs = self.llm.complete(
                system_prompt=(
                    "Summarize this portion of a meeting transcript. "
                    "Extract key points and topics discussed. Return valid JSON."
                ),
                user_prompt=chunk,
                response_model=ChunkSummary,
            )
            chunk_summaries.append(cs)

        # Reduce phase: combine chunk summaries into final summary
        combined = "\n\n".join(
            f"Part {i + 1}:\n- Topics: {', '.join(cs.topics)}\n"
            f"- Key points: {'; '.join(cs.key_points)}\n- Summary: {cs.summary}"
            for i, cs in enumerate(chunk_summaries)
        )

        user_prompt = f"Combined meeting notes from {len(chunks)} parts:\n\n{combined}"
        if context:
            user_prompt += f"\n\nAdditional context:\n{context}"

        result = self.llm.complete(
            system_prompt=SUMMARY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=MeetingSummary,
        )
        result.participant_count = len(transcript.speakers)
        result.duration_minutes = round(transcript.duration_seconds / 60.0, 1)
        return result

    def _build_context(
        self,
        decisions: list[Decision] | None,
        action_items: list[ActionItem] | None,
    ) -> str:
        """Build additional context string from decisions and action items."""
        parts = []
        if decisions:
            parts.append("Decisions made:")
            for d in decisions:
                parts.append(f"  - {d.text}")
        if action_items:
            parts.append("Action items:")
            for a in action_items:
                assignee = a.assignee or "unassigned"
                parts.append(f"  - [{assignee}] {a.task}")
        return "\n".join(parts)
