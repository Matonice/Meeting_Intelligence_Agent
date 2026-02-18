"""Transcript cleaning: disfluency removal and text normalization."""

from __future__ import annotations

import re

from ..models.transcript import Transcript, TranscriptSegment
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TranscriptCleaner:
    """Cleans raw ASR output by removing disfluencies and normalizing text."""

    DISFLUENCIES = re.compile(
        r"\b(uh+|um+|er+|ah+|hmm+|huh|"
        r"like,?\s+|you know,?\s+|I mean,?\s+|"
        r"sort of,?\s+|kind of,?\s+|"
        r"basically,?\s+|actually,?\s+)\b",
        re.IGNORECASE,
    )

    REPEATED_WORDS = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)

    MULTI_SPACE = re.compile(r"\s{2,}")

    def clean(self, transcript: Transcript) -> Transcript:
        """Apply all cleaning steps to transcript.

        Returns a new Transcript with cleaned segment texts.
        """
        cleaned_segments = []
        for seg in transcript.segments:
            text = seg.text
            text = self.remove_disfluencies(text)
            text = self.remove_repeated_words(text)
            text = self.normalize_whitespace(text)
            text = self.capitalize_sentences(text)

            if text.strip():
                cleaned_segments.append(seg.model_copy(update={"text": text.strip()}))

        logger.info(
            f"Cleaning: {len(transcript.segments)} → {len(cleaned_segments)} segments"
        )

        return transcript.model_copy(update={"segments": cleaned_segments})

    def remove_disfluencies(self, text: str) -> str:
        """Remove filler words and verbal disfluencies."""
        return self.DISFLUENCIES.sub("", text)

    def remove_repeated_words(self, text: str) -> str:
        """Remove immediately repeated words ('the the' → 'the')."""
        return self.REPEATED_WORDS.sub(r"\1", text)

    def normalize_whitespace(self, text: str) -> str:
        """Collapse multiple spaces into one."""
        return self.MULTI_SPACE.sub(" ", text)

    def capitalize_sentences(self, text: str) -> str:
        """Capitalize the first letter of each sentence."""
        if not text:
            return text
        # Capitalize after sentence-ending punctuation
        result = re.sub(
            r"([.!?]\s+)([a-z])",
            lambda m: m.group(1) + m.group(2).upper(),
            text,
        )
        # Capitalize first character
        if result and result[0].islower():
            result = result[0].upper() + result[1:]
        return result
