"""Tests for audio loading and normalization."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from meeting_intelligence.audio.loader import (
    SUPPORTED_FORMATS,
    AudioLoadError,
    get_audio_metadata,
    load_audio,
    validate_audio_file,
)
from meeting_intelligence.models.audio import AudioData, AudioMetadata


class TestValidateAudioFile:
    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(AudioLoadError, match="File not found"):
            validate_audio_file(tmp_path / "nonexistent.wav")

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.wav"
        f.write_bytes(b"")
        with pytest.raises(AudioLoadError, match="File is empty"):
            validate_audio_file(f)

    def test_unsupported_format_raises(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("not audio")
        with pytest.raises(AudioLoadError, match="Unsupported format"):
            validate_audio_file(f)

    def test_supported_format_passes(self, tmp_path):
        f = tmp_path / "audio.wav"
        f.write_bytes(b"\x00" * 100)
        # Should not raise
        validate_audio_file(f)

    def test_case_insensitive_extension(self, tmp_path):
        f = tmp_path / "audio.WAV"
        f.write_bytes(b"\x00" * 100)
        validate_audio_file(f)

    def test_all_supported_formats(self):
        expected = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".webm"}
        assert SUPPORTED_FORMATS == expected


class TestLoadAudio:
    @patch("meeting_intelligence.audio.loader.AudioSegment")
    def test_returns_audio_data(self, mock_audio_seg_cls, tmp_path):
        # Create a dummy file
        f = tmp_path / "test.wav"
        f.write_bytes(b"\x00" * 100)

        # Mock pydub AudioSegment
        mock_audio = MagicMock()
        mock_audio.frame_rate = 44100
        mock_audio.channels = 2
        mock_audio.set_channels.return_value = mock_audio
        mock_audio.set_frame_rate.return_value = mock_audio
        mock_audio.set_sample_width.return_value = mock_audio
        mock_audio.__len__ = lambda self: 3000  # 3 seconds in ms
        mock_audio.get_array_of_samples.return_value = [0] * 48000  # 3s at 16kHz

        mock_audio_seg_cls.from_file.return_value = mock_audio

        result = load_audio(f, target_sample_rate=16000)

        assert isinstance(result, AudioData)
        assert result.sample_rate == 16000
        assert result.samples.dtype == np.float32
        assert result.metadata.file_path == f

    @patch("meeting_intelligence.audio.loader.AudioSegment")
    def test_normalizes_to_float32_range(self, mock_audio_seg_cls, tmp_path):
        f = tmp_path / "test.wav"
        f.write_bytes(b"\x00" * 100)

        mock_audio = MagicMock()
        mock_audio.frame_rate = 16000
        mock_audio.channels = 1
        mock_audio.set_channels.return_value = mock_audio
        mock_audio.set_frame_rate.return_value = mock_audio
        mock_audio.set_sample_width.return_value = mock_audio
        mock_audio.__len__ = lambda self: 1000
        # Max 16-bit value
        mock_audio.get_array_of_samples.return_value = [32767, -32768, 0]

        mock_audio_seg_cls.from_file.return_value = mock_audio

        result = load_audio(f)

        assert result.samples[0] == pytest.approx(32767 / 32768.0)
        assert result.samples[1] == pytest.approx(-32768 / 32768.0)
        assert result.samples[2] == 0.0

    @patch("meeting_intelligence.audio.loader.AudioSegment")
    def test_metadata_captures_original_properties(self, mock_audio_seg_cls, tmp_path):
        f = tmp_path / "test.mp3"
        f.write_bytes(b"\x00" * 500)

        mock_audio = MagicMock()
        mock_audio.frame_rate = 44100
        mock_audio.channels = 2
        mock_audio.set_channels.return_value = mock_audio
        mock_audio.set_frame_rate.return_value = mock_audio
        mock_audio.set_sample_width.return_value = mock_audio
        mock_audio.__len__ = lambda self: 5000
        mock_audio.get_array_of_samples.return_value = [0] * 80000

        mock_audio_seg_cls.from_file.return_value = mock_audio

        result = load_audio(f)

        assert result.metadata.sample_rate == 44100  # original
        assert result.metadata.channels == 2  # original
        assert result.metadata.format == "mp3"
        assert result.metadata.file_size_bytes == 500

    @patch("meeting_intelligence.audio.loader.AudioSegment")
    def test_custom_sample_rate(self, mock_audio_seg_cls, tmp_path):
        f = tmp_path / "test.wav"
        f.write_bytes(b"\x00" * 100)

        mock_audio = MagicMock()
        mock_audio.frame_rate = 44100
        mock_audio.channels = 1
        mock_audio.set_channels.return_value = mock_audio
        mock_audio.set_frame_rate.return_value = mock_audio
        mock_audio.set_sample_width.return_value = mock_audio
        mock_audio.__len__ = lambda self: 1000
        mock_audio.get_array_of_samples.return_value = [0] * 8000

        mock_audio_seg_cls.from_file.return_value = mock_audio

        result = load_audio(f, target_sample_rate=8000)

        assert result.sample_rate == 8000
        mock_audio.set_frame_rate.assert_called_with(8000)

    def test_load_nonexistent_raises(self):
        with pytest.raises(AudioLoadError, match="File not found"):
            load_audio(Path("/nonexistent/file.wav"))


class TestGetAudioMetadata:
    @patch("meeting_intelligence.audio.loader.AudioSegment")
    def test_returns_metadata(self, mock_audio_seg_cls, tmp_path):
        f = tmp_path / "test.wav"
        f.write_bytes(b"\x00" * 200)

        mock_audio = MagicMock()
        mock_audio.frame_rate = 48000
        mock_audio.channels = 1
        mock_audio.__len__ = lambda self: 2000

        mock_audio_seg_cls.from_file.return_value = mock_audio

        meta = get_audio_metadata(f)

        assert isinstance(meta, AudioMetadata)
        assert meta.sample_rate == 48000
        assert meta.duration_seconds == 2.0
        assert meta.file_size_bytes == 200
