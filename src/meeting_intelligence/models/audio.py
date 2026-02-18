"""Audio-related data models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field


class AudioMetadata(BaseModel):
    """Metadata extracted from an audio file."""

    file_path: Path
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str
    file_size_bytes: int


class AudioData(BaseModel):
    """Normalized audio ready for processing. Always mono, 16kHz, float32."""

    model_config = {"arbitrary_types_allowed": True}

    samples: NDArray[np.float32] = Field(exclude=True)
    sample_rate: int = 16000
    duration_seconds: float
    metadata: AudioMetadata


class SpeechSegment(BaseModel):
    """A segment of detected speech from VAD."""

    start: float
    end: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start
