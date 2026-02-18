"""Speech-to-text using faster-whisper."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from ..config import ASRConfig
from ..models.audio import AudioData
from ..models.transcript import TranscriptSegment, Word
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ASREngine:
    """Transcribes audio using faster-whisper (CTranslate2-optimized Whisper)."""

    def __init__(self, config: ASRConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = device
        self._model: WhisperModel | None = None

    def _load_model(self) -> None:
        """Lazy-load the Whisper model."""
        if self._model is not None:
            return
        logger.info(f"Loading Whisper model: {self.config.model_size} ({self.config.compute_type})")
        self._model = WhisperModel(
            self.config.model_size,
            device=self.device,
            compute_type=self.config.compute_type,
        )

    def transcribe(
        self,
        audio: AudioData,
        language: str | None = None,
    ) -> tuple[list[TranscriptSegment], str]:
        """Transcribe audio to text with word-level timestamps.

        Args:
            audio: Normalized audio data (mono, 16kHz).
            language: Language code, or None for auto-detection.

        Returns:
            Tuple of (transcript segments, detected language code).
        """
        self._load_model()

        logger.info("Transcribing audio...")

        # faster-whisper accepts numpy arrays directly
        segments_gen, info = self._model.transcribe(
            audio.samples,
            beam_size=self.config.beam_size,
            word_timestamps=self.config.word_timestamps,
            language=language,
        )

        detected_lang = info.language
        logger.info(f"Detected language: {detected_lang} ({info.language_probability:.0%})")

        # Consume the generator
        transcript_segments = []
        for segment in segments_gen:
            words = []
            if segment.words:
                words = [
                    Word(
                        text=w.word.strip(),
                        start=w.start,
                        end=w.end,
                        confidence=w.probability,
                    )
                    for w in segment.words
                ]

            transcript_segments.append(
                TranscriptSegment(
                    speaker="",  # filled in by assembler
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                    words=words,
                    confidence=segment.avg_logprob,
                )
            )

        logger.info(f"ASR: {len(transcript_segments)} segments transcribed")
        return transcript_segments, detected_lang

    def transcribe_segment(
        self,
        audio: AudioData,
        start: float,
        end: float,
        language: str | None = None,
    ) -> TranscriptSegment | None:
        """Transcribe a specific time range of the audio."""
        self._load_model()

        start_sample = int(start * audio.sample_rate)
        end_sample = int(end * audio.sample_rate)
        chunk = audio.samples[start_sample:end_sample]

        if len(chunk) == 0:
            return None

        segments_gen, info = self._model.transcribe(
            chunk,
            beam_size=self.config.beam_size,
            word_timestamps=self.config.word_timestamps,
            language=language,
        )

        text_parts = []
        all_words = []
        for seg in segments_gen:
            text_parts.append(seg.text.strip())
            if seg.words:
                all_words.extend(
                    Word(
                        text=w.word.strip(),
                        start=w.start + start,
                        end=w.end + start,
                        confidence=w.probability,
                    )
                    for w in seg.words
                )

        full_text = " ".join(text_parts)
        if not full_text:
            return None

        return TranscriptSegment(
            speaker="",
            start=start,
            end=end,
            text=full_text,
            words=all_words,
        )
