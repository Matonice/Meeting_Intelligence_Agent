"""Q&A over meeting transcripts."""

from __future__ import annotations

from ..models.meeting import QAResponse
from ..models.transcript import Transcript
from ..utils.logging import get_logger
from .llm_client import LLMClient

logger = get_logger(__name__)

QA_SYSTEM_PROMPT = """You are a meeting assistant. Answer questions about the meeting based solely on the transcript provided.

Rules:
- Only answer based on information in the transcript
- If the answer is not in the transcript, clearly say so
- Cite specific speaker statements when possible
- Be concise and direct
- Include the confidence level (0.0-1.0) of your answer

Return valid JSON with: question, answer, relevant_segments (list of segment indices), confidence."""


class MeetingQA:
    """Answers questions about meetings using the transcript as context."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client
        self._transcript: Transcript | None = None

    def load_transcript(self, transcript: Transcript) -> None:
        """Set the transcript context for Q&A."""
        self._transcript = transcript

    def ask(self, question: str, transcript: Transcript | None = None) -> QAResponse:
        """Answer a question about the meeting.

        Args:
            question: The question to answer.
            transcript: Transcript to use (or uses pre-loaded one).

        Returns:
            QAResponse with answer and supporting evidence.
        """
        t = transcript or self._transcript
        if t is None:
            raise ValueError("No transcript loaded. Call load_transcript() or pass transcript.")

        logger.info(f"Q&A: {question}")

        # Build numbered transcript for segment reference
        numbered_lines = []
        for i, seg in enumerate(t.segments):
            numbered_lines.append(f"[{i}] [{seg.start:.1f}s] {seg.speaker}: {seg.text}")
        transcript_text = "\n".join(numbered_lines)

        # Handle long transcripts
        token_count = self.llm.count_tokens(transcript_text)
        if token_count > 6000:
            chunks = self.llm.chunk_transcript(transcript_text, max_tokens=5500)
            transcript_text = chunks[-1]  # Use most recent portion
            if len(chunks) > 1:
                transcript_text = (
                    "[Earlier discussion omitted]\n...\n\n" + transcript_text
                )

        result = self.llm.complete(
            system_prompt=QA_SYSTEM_PROMPT,
            user_prompt=f"Transcript:\n{transcript_text}\n\nQuestion: {question}",
            response_model=QAResponse,
        )

        # Ensure question is echoed back
        result.question = question
        return result
