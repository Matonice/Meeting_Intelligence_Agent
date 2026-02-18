"""Decision and action item extraction from meeting transcripts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..models.meeting import ActionItem, Decision, Priority
from ..models.transcript import Transcript
from ..utils.logging import get_logger
from .llm_client import LLMClient

logger = get_logger(__name__)


class DecisionList(BaseModel):
    """Structured output model for decision extraction."""

    decisions: list[Decision] = Field(default_factory=list)


class ActionItemList(BaseModel):
    """Structured output model for action item extraction."""

    action_items: list[ActionItem] = Field(default_factory=list)


DECISION_SYSTEM_PROMPT = """You are a meeting analyst. Extract all decisions made during this meeting.

A decision is a commitment, agreement, or resolution to a specific course of action.
Examples:
- "We'll deploy on Friday" → Decision
- "Let's go with option B" → Decision
- "We agreed to postpone the launch" → Decision

For each decision, provide:
- id: sequential number starting from 1
- text: the decision statement
- context: brief surrounding context from the transcript
- participants: list of speaker IDs involved in the decision
- confidence: 0.0-1.0 how confident you are this is a real decision

Only extract clear decisions, not suggestions or questions. Return valid JSON."""

ACTION_SYSTEM_PROMPT = """You are a meeting analyst. Extract all action items (tasks) from this meeting.

An action item is a specific task assigned to or taken on by someone.
Examples:
- "John, send the report by Friday" → Task(assignee=John, task=send the report, deadline=Friday)
- "I'll update the documentation" → Task(assignee=speaker, task=update documentation)

For each action item, provide:
- id: sequential number starting from 1
- task: clear description of what needs to be done
- assignee: who is responsible (use speaker ID if name unknown, null if unassigned)
- deadline: mentioned deadline if any (null otherwise)
- priority: high/medium/low based on urgency cues
- context: brief surrounding context
- confidence: 0.0-1.0 how confident you are this is a real action item

Only extract concrete tasks, not vague intentions. Return valid JSON."""


class MeetingExtractor:
    """Extracts decisions and action items from meeting transcripts."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def extract_decisions(self, transcript: Transcript) -> list[Decision]:
        """Extract decisions from the transcript using GPT-4."""
        logger.info("Extracting decisions...")
        transcript_text = self._prepare_transcript(transcript)

        result = self.llm.complete(
            system_prompt=DECISION_SYSTEM_PROMPT,
            user_prompt=f"Meeting transcript:\n\n{transcript_text}",
            response_model=DecisionList,
        )

        logger.info(f"Extracted {len(result.decisions)} decisions")
        return result.decisions

    def extract_action_items(self, transcript: Transcript) -> list[ActionItem]:
        """Extract action items from the transcript using GPT-4."""
        logger.info("Extracting action items...")
        transcript_text = self._prepare_transcript(transcript)

        result = self.llm.complete(
            system_prompt=ACTION_SYSTEM_PROMPT,
            user_prompt=f"Meeting transcript:\n\n{transcript_text}",
            response_model=ActionItemList,
        )

        logger.info(f"Extracted {len(result.action_items)} action items")
        return result.action_items

    def _prepare_transcript(self, transcript: Transcript) -> str:
        """Format transcript for LLM, with chunking if needed."""
        text = transcript.speaker_text
        token_count = self.llm.count_tokens(text)

        if token_count > 6000:
            logger.debug(f"Transcript is {token_count} tokens, will use last portion")
            # For extraction, prioritize the full transcript but truncate from start if needed
            chunks = self.llm.chunk_transcript(text, max_tokens=6000)
            if len(chunks) > 1:
                # Use last chunk + summary of earlier parts
                text = (
                    "[Earlier discussion omitted for length]\n...\n\n"
                    + chunks[-1]
                )

        return text
