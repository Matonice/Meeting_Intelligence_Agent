"""OpenAI GPT-4 client wrapper with structured output and retry logic."""

from __future__ import annotations

import json
from typing import TypeVar

import tiktoken
from openai import OpenAI
from pydantic import BaseModel

from ..config import LLMConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Centralized LLM client for all meeting understanding tasks."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.client = OpenAI(api_key=config.api_key)
        self.model = config.model
        self._encoder: tiktoken.Encoding | None = None

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T] | None = None,
    ) -> str | T:
        """Send a chat completion request.

        If response_model is provided, parse into structured Pydantic model.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if response_model is not None:
            return self._complete_structured(messages, response_model)
        else:
            return self._complete_text(messages)

    def _complete_text(self, messages: list[dict]) -> str:
        """Plain text completion."""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return completion.choices[0].message.content or ""

    def _complete_structured(self, messages: list[dict], response_model: type[T]) -> T:
        """Structured output completion with Pydantic model parsing.

        Tries in order:
        1. Native structured output (gpt-4o+)
        2. JSON mode (gpt-4-turbo, gpt-4o)
        3. Plain text with JSON instruction (any model)
        """
        # 1. Try native structured output (works with gpt-4o and later)
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is not None:
                return parsed
        except Exception:
            logger.debug("Structured output not available, trying JSON mode")

        # Build messages with schema instruction for fallbacks
        schema_instruction = (
            "\n\nYou MUST respond with valid JSON and nothing else. "
            "Match this schema:\n" + json.dumps(response_model.model_json_schema(), indent=2)
        )
        fallback_messages = messages.copy()
        fallback_messages[0] = {
            "role": "system",
            "content": messages[0]["content"] + schema_instruction,
        }

        # 2. Try JSON mode (works with gpt-4-turbo, gpt-4o)
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=fallback_messages,
                response_format={"type": "json_object"},
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            raw = completion.choices[0].message.content or "{}"
            return response_model.model_validate_json(raw)
        except Exception:
            logger.debug("JSON mode not available, using plain text fallback")

        # 3. Plain text fallback (works with any model)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=fallback_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        raw = completion.choices[0].message.content or "{}"
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        return response_model.model_validate_json(raw)

    def count_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken."""
        if self._encoder is None:
            try:
                self._encoder = tiktoken.encoding_for_model(self.model)
            except KeyError:
                self._encoder = tiktoken.get_encoding("cl100k_base")
        return len(self._encoder.encode(text))

    def chunk_transcript(self, text: str, max_tokens: int = 6000) -> list[str]:
        """Split text into chunks that fit within token limits.

        Splits on paragraph/line boundaries to keep segments intact.
        """
        lines = text.split("\n")
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for line in lines:
            line_tokens = self.count_tokens(line)
            if current_tokens + line_tokens > max_tokens and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            current_chunk.append(line)
            current_tokens += line_tokens

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks
