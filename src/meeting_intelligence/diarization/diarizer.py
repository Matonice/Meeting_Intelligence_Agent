"""Speaker diarization using pyannote.audio."""

from __future__ import annotations

import torch
from pyannote.audio import Pipeline

from ..config import DiarizationConfig
from ..models.audio import AudioData
from ..models.transcript import DiarizationSegment
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SpeakerDiarizer:
    """Determines who spoke when using pyannote speaker diarization."""

    def __init__(self, config: DiarizationConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = device
        self._pipeline: Pipeline | None = None

    def _load_pipeline(self) -> None:
        """Lazy-load the pyannote diarization pipeline."""
        if self._pipeline is not None:
            return
        logger.info(f"Loading diarization model: {self.config.model}")
        self._pipeline = Pipeline.from_pretrained(
            self.config.model,
            token=self.config.huggingface_token or None,
        )
        if self.device == "cuda" and torch.cuda.is_available():
            self._pipeline.to(torch.device("cuda"))

    def diarize(
        self,
        audio: AudioData,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[DiarizationSegment]:
        """Run speaker diarization on audio.

        Args:
            audio: Normalized audio data.
            num_speakers: Exact number of speakers (if known).
            min_speakers: Minimum expected speakers.
            max_speakers: Maximum expected speakers.

        Returns:
            List of DiarizationSegment with speaker labels and time ranges.
        """
        self._load_pipeline()

        # Pass audio directly as waveform tensor
        waveform = torch.from_numpy(audio.samples).unsqueeze(0)  # (1, num_samples)
        audio_input = {"waveform": waveform, "sample_rate": audio.sample_rate}

        # Build kwargs
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers is not None or self.config.min_speakers is not None:
                kwargs["min_speakers"] = min_speakers or self.config.min_speakers
            if max_speakers is not None or self.config.max_speakers is not None:
                kwargs["max_speakers"] = max_speakers or self.config.max_speakers

        logger.info("Running speaker diarization...")
        output = self._pipeline(audio_input, **kwargs)

        # pyannote.audio v4.0+ returns DiarizeOutput; extract the Annotation
        annotation = getattr(output, "speaker_diarization", output)

        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(
                DiarizationSegment(
                    speaker=speaker,
                    start=turn.start,
                    end=turn.end,
                )
            )

        speakers = set(s.speaker for s in segments)
        logger.info(f"Diarization: {len(speakers)} speakers, {len(segments)} turns")

        return segments

    def get_speaker_stats(self, segments: list[DiarizationSegment]) -> dict[str, float]:
        """Return total speaking time per speaker."""
        stats: dict[str, float] = {}
        for seg in segments:
            stats[seg.speaker] = stats.get(seg.speaker, 0.0) + (seg.end - seg.start)
        return stats
