"""Voice Activity Detection using Silero VAD."""

from __future__ import annotations

import torch

from ..config import VADConfig
from ..models.audio import AudioData, SpeechSegment
from ..utils.logging import get_logger

logger = get_logger(__name__)


class VADProcessor:
    """Detects speech regions in audio using Silero VAD."""

    def __init__(self, config: VADConfig) -> None:
        self.config = config
        self._model = None
        self._utils = None

    def _load_model(self) -> None:
        """Lazy-load the Silero VAD model."""
        if self._model is not None:
            return
        logger.info("Loading Silero VAD model...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model = model
        self._utils = utils

    def detect_speech(self, audio: AudioData) -> list[SpeechSegment]:
        """Run VAD on audio, returning speech segments.

        Args:
            audio: Normalized audio data (mono, 16kHz).

        Returns:
            List of SpeechSegment with start/end times in seconds.
        """
        self._load_model()

        # Silero VAD expects a torch tensor
        waveform = torch.from_numpy(audio.samples)

        get_speech_timestamps = self._utils[0]
        timestamps = get_speech_timestamps(
            waveform,
            self._model,
            threshold=self.config.threshold,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            sampling_rate=audio.sample_rate,
            return_seconds=True,
        )

        segments = [SpeechSegment(start=ts["start"], end=ts["end"]) for ts in timestamps]

        total_speech = sum(s.duration for s in segments)
        ratio = total_speech / audio.duration_seconds if audio.duration_seconds > 0 else 0
        logger.info(
            f"VAD: {len(segments)} speech segments, "
            f"{total_speech:.1f}s speech ({ratio:.0%} of audio)"
        )

        return segments

    def get_speech_ratio(self, segments: list[SpeechSegment], duration: float) -> float:
        """Calculate what percentage of audio contains speech."""
        if duration <= 0:
            return 0.0
        total = sum(s.duration for s in segments)
        return total / duration
