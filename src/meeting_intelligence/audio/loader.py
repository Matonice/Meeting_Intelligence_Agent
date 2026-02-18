"""Audio file loading and normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydub import AudioSegment

from ..models.audio import AudioData, AudioMetadata
from ..utils.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".webm"}


class AudioLoadError(Exception):
    """Raised when an audio file cannot be loaded."""


def validate_audio_file(file_path: Path) -> None:
    """Validate that the audio file exists and has a supported format."""
    if not file_path.exists():
        raise AudioLoadError(f"File not found: {file_path}")
    if file_path.stat().st_size == 0:
        raise AudioLoadError(f"File is empty: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise AudioLoadError(
            f"Unsupported format '{file_path.suffix}'. Supported: {SUPPORTED_FORMATS}"
        )


def get_audio_metadata(file_path: Path) -> AudioMetadata:
    """Extract metadata without fully loading audio into memory."""
    validate_audio_file(file_path)
    audio = AudioSegment.from_file(str(file_path))
    return AudioMetadata(
        file_path=file_path,
        duration_seconds=len(audio) / 1000.0,
        sample_rate=audio.frame_rate,
        channels=audio.channels,
        format=file_path.suffix.lstrip("."),
        file_size_bytes=file_path.stat().st_size,
    )


def load_audio(file_path: Path, target_sample_rate: int = 16000) -> AudioData:
    """Load an audio file and normalize to mono 16kHz float32.

    Args:
        file_path: Path to the audio file.
        target_sample_rate: Target sample rate (default 16kHz for speech models).

    Returns:
        AudioData with normalized samples.
    """
    file_path = Path(file_path)
    validate_audio_file(file_path)

    logger.info(f"Loading audio: {file_path.name}")

    # Load with pydub
    audio = AudioSegment.from_file(str(file_path))

    # Capture original metadata
    metadata = AudioMetadata(
        file_path=file_path,
        duration_seconds=len(audio) / 1000.0,
        sample_rate=audio.frame_rate,
        channels=audio.channels,
        format=file_path.suffix.lstrip("."),
        file_size_bytes=file_path.stat().st_size,
    )

    logger.debug(
        f"Original: {metadata.sample_rate}Hz, {metadata.channels}ch, "
        f"{metadata.duration_seconds:.1f}s"
    )

    # Normalize: mono, target sample rate, 16-bit
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(target_sample_rate)
    audio = audio.set_sample_width(2)  # 16-bit

    # Convert to numpy float32 array
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples /= 32768.0  # normalize 16-bit to [-1, 1]

    duration = len(samples) / target_sample_rate
    logger.info(f"Loaded: {duration:.1f}s, {target_sample_rate}Hz mono")

    return AudioData(
        samples=samples,
        sample_rate=target_sample_rate,
        duration_seconds=duration,
        metadata=metadata,
    )
