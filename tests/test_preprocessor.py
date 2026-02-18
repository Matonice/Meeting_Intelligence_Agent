"""Tests for Voice Activity Detection preprocessor."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meeting_intelligence.audio.preprocessor import VADProcessor
from meeting_intelligence.config import VADConfig
from meeting_intelligence.models.audio import AudioData, AudioMetadata, SpeechSegment


@pytest.fixture
def vad_config():
    return VADConfig(threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=100)


@pytest.fixture
def mock_vad_model():
    """Create a mock Silero VAD model and utils."""
    model = MagicMock()
    get_speech_timestamps = MagicMock()
    utils = (get_speech_timestamps, MagicMock(), MagicMock(), MagicMock())
    return model, utils, get_speech_timestamps


class TestVADProcessor:
    @patch("meeting_intelligence.audio.preprocessor.torch")
    def test_detect_speech_returns_segments(
        self, mock_torch, vad_config, mock_vad_model, sample_audio
    ):
        model, utils, get_speech_timestamps = mock_vad_model
        mock_torch.hub.load.return_value = (model, utils)
        mock_torch.from_numpy.return_value = MagicMock()

        get_speech_timestamps.return_value = [
            {"start": 0.5, "end": 1.5},
            {"start": 2.0, "end": 2.8},
        ]

        vad = VADProcessor(vad_config)
        segments = vad.detect_speech(sample_audio)

        assert len(segments) == 2
        assert segments[0].start == 0.5
        assert segments[0].end == 1.5
        assert segments[1].start == 2.0
        assert segments[1].end == 2.8

    @patch("meeting_intelligence.audio.preprocessor.torch")
    def test_detect_speech_no_speech_found(
        self, mock_torch, vad_config, mock_vad_model, sample_audio
    ):
        model, utils, get_speech_timestamps = mock_vad_model
        mock_torch.hub.load.return_value = (model, utils)
        mock_torch.from_numpy.return_value = MagicMock()

        get_speech_timestamps.return_value = []

        vad = VADProcessor(vad_config)
        segments = vad.detect_speech(sample_audio)

        assert segments == []

    @patch("meeting_intelligence.audio.preprocessor.torch")
    def test_model_loaded_only_once(
        self, mock_torch, vad_config, mock_vad_model, sample_audio
    ):
        model, utils, get_speech_timestamps = mock_vad_model
        mock_torch.hub.load.return_value = (model, utils)
        mock_torch.from_numpy.return_value = MagicMock()
        get_speech_timestamps.return_value = []

        vad = VADProcessor(vad_config)
        vad.detect_speech(sample_audio)
        vad.detect_speech(sample_audio)

        # torch.hub.load called only once despite two detect_speech calls
        mock_torch.hub.load.assert_called_once()

    @patch("meeting_intelligence.audio.preprocessor.torch")
    def test_passes_config_to_vad(
        self, mock_torch, mock_vad_model, sample_audio
    ):
        config = VADConfig(threshold=0.7, min_speech_duration_ms=500, min_silence_duration_ms=200)
        model, utils, get_speech_timestamps = mock_vad_model
        mock_torch.hub.load.return_value = (model, utils)
        mock_torch.from_numpy.return_value = MagicMock()
        get_speech_timestamps.return_value = []

        vad = VADProcessor(config)
        vad.detect_speech(sample_audio)

        call_kwargs = get_speech_timestamps.call_args[1]
        assert call_kwargs["threshold"] == 0.7
        assert call_kwargs["min_speech_duration_ms"] == 500
        assert call_kwargs["min_silence_duration_ms"] == 200
        assert call_kwargs["return_seconds"] is True

    def test_get_speech_ratio_normal(self):
        segments = [
            SpeechSegment(start=0.0, end=2.0),
            SpeechSegment(start=3.0, end=4.0),
        ]
        vad = VADProcessor(VADConfig())
        ratio = vad.get_speech_ratio(segments, 10.0)
        assert ratio == pytest.approx(0.3)

    def test_get_speech_ratio_zero_duration(self):
        vad = VADProcessor(VADConfig())
        assert vad.get_speech_ratio([], 0.0) == 0.0

    def test_get_speech_ratio_empty_segments(self):
        vad = VADProcessor(VADConfig())
        assert vad.get_speech_ratio([], 5.0) == 0.0

    def test_get_speech_ratio_full_speech(self):
        segments = [SpeechSegment(start=0.0, end=10.0)]
        vad = VADProcessor(VADConfig())
        assert vad.get_speech_ratio(segments, 10.0) == pytest.approx(1.0)
