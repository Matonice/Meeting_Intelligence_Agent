"""Tests for ASR (speech-to-text) engine."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meeting_intelligence.config import ASRConfig
from meeting_intelligence.models.audio import AudioData, AudioMetadata
from meeting_intelligence.transcription.asr import ASREngine
from pathlib import Path


@pytest.fixture
def asr_config():
    return ASRConfig(
        model_size="tiny",
        compute_type="int8",
        beam_size=3,
        word_timestamps=True,
    )


def _make_mock_word(word, start, end, probability=0.9):
    """Create a mock faster-whisper Word object."""
    w = MagicMock()
    w.word = word
    w.start = start
    w.end = end
    w.probability = probability
    return w


def _make_mock_segment(start, end, text, words=None, avg_logprob=-0.3):
    """Create a mock faster-whisper Segment object."""
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    seg.words = words or []
    seg.avg_logprob = avg_logprob
    return seg


def _make_mock_info(language="en", probability=0.98):
    info = MagicMock()
    info.language = language
    info.language_probability = probability
    return info


class TestASREngine:
    @patch("meeting_intelligence.transcription.asr.WhisperModel")
    def test_transcribe_returns_segments(
        self, mock_whisper_cls, asr_config, sample_audio
    ):
        words = [
            _make_mock_word(" Hello", 0.0, 0.5),
            _make_mock_word(" world", 0.6, 1.0),
        ]
        mock_seg = _make_mock_segment(0.0, 1.0, " Hello world", words)
        mock_info = _make_mock_info("en", 0.98)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_seg], mock_info)
        mock_whisper_cls.return_value = mock_model

        engine = ASREngine(asr_config)
        segments, lang = engine.transcribe(sample_audio)

        assert lang == "en"
        assert len(segments) == 1
        assert segments[0].text == "Hello world"
        assert segments[0].speaker == ""  # not yet assigned
        assert len(segments[0].words) == 2
        assert segments[0].words[0].text == "Hello"
        assert segments[0].words[1].text == "world"

    @patch("meeting_intelligence.transcription.asr.WhisperModel")
    def test_transcribe_multiple_segments(
        self, mock_whisper_cls, asr_config, sample_audio
    ):
        seg1 = _make_mock_segment(0.0, 2.0, " First segment")
        seg2 = _make_mock_segment(2.5, 4.0, " Second segment")
        mock_info = _make_mock_info("en")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], mock_info)
        mock_whisper_cls.return_value = mock_model

        engine = ASREngine(asr_config)
        segments, _ = engine.transcribe(sample_audio)

        assert len(segments) == 2
        assert segments[0].text == "First segment"
        assert segments[1].text == "Second segment"

    @patch("meeting_intelligence.transcription.asr.WhisperModel")
    def test_transcribe_no_words(
        self, mock_whisper_cls, asr_config, sample_audio
    ):
        seg = _make_mock_segment(0.0, 1.0, " Hello", words=None)
        mock_info = _make_mock_info()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], mock_info)
        mock_whisper_cls.return_value = mock_model

        engine = ASREngine(asr_config)
        segments, _ = engine.transcribe(sample_audio)

        assert len(segments) == 1
        assert segments[0].words == []

    @patch("meeting_intelligence.transcription.asr.WhisperModel")
    def test_transcribe_empty_result(
        self, mock_whisper_cls, asr_config, sample_audio
    ):
        mock_info = _make_mock_info()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], mock_info)
        mock_whisper_cls.return_value = mock_model

        engine = ASREngine(asr_config)
        segments, _ = engine.transcribe(sample_audio)

        assert segments == []

    @patch("meeting_intelligence.transcription.asr.WhisperModel")
    def test_language_autodetection(
        self, mock_whisper_cls, asr_config, sample_audio
    ):
        mock_info = _make_mock_info("fr", 0.95)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], mock_info)
        mock_whisper_cls.return_value = mock_model

        engine = ASREngine(asr_config)
        _, lang = engine.transcribe(sample_audio, language=None)

        assert lang == "fr"

    @patch("meeting_intelligence.transcription.asr.WhisperModel")
    def test_model_loaded_once(
        self, mock_whisper_cls, asr_config, sample_audio
    ):
        mock_info = _make_mock_info()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], mock_info)
        mock_whisper_cls.return_value = mock_model

        engine = ASREngine(asr_config)
        engine.transcribe(sample_audio)
        engine.transcribe(sample_audio)

        mock_whisper_cls.assert_called_once()

    @patch("meeting_intelligence.transcription.asr.WhisperModel")
    def test_transcribe_passes_config(
        self, mock_whisper_cls, asr_config, sample_audio
    ):
        mock_info = _make_mock_info()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], mock_info)
        mock_whisper_cls.return_value = mock_model

        engine = ASREngine(asr_config)
        engine.transcribe(sample_audio, language="de")

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["beam_size"] == 3
        assert call_kwargs["word_timestamps"] is True
        assert call_kwargs["language"] == "de"

    @patch("meeting_intelligence.transcription.asr.WhisperModel")
    def test_transcribe_segment_returns_text(
        self, mock_whisper_cls, asr_config, sample_audio
    ):
        seg = _make_mock_segment(0.0, 0.5, " Hello")
        mock_info = _make_mock_info()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], mock_info)
        mock_whisper_cls.return_value = mock_model

        engine = ASREngine(asr_config)
        result = engine.transcribe_segment(sample_audio, start=0.5, end=1.5)

        assert result is not None
        assert result.text == "Hello"
        assert result.start == 0.5
        assert result.end == 1.5

    @patch("meeting_intelligence.transcription.asr.WhisperModel")
    def test_transcribe_segment_empty_returns_none(
        self, mock_whisper_cls, asr_config, sample_audio
    ):
        mock_info = _make_mock_info()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], mock_info)
        mock_whisper_cls.return_value = mock_model

        engine = ASREngine(asr_config)
        result = engine.transcribe_segment(sample_audio, start=0.0, end=0.5)

        assert result is None

    @patch("meeting_intelligence.transcription.asr.WhisperModel")
    def test_transcribe_segment_zero_length_returns_none(
        self, mock_whisper_cls, asr_config
    ):
        mock_whisper_cls.return_value = MagicMock()

        # Audio with exactly 0 samples for the slice
        samples = np.zeros(16000, dtype=np.float32)
        audio = AudioData(
            samples=samples,
            sample_rate=16000,
            duration_seconds=1.0,
            metadata=AudioMetadata(
                file_path=Path("/tmp/t.wav"),
                duration_seconds=1.0, sample_rate=16000,
                channels=1, format="wav", file_size_bytes=32000,
            ),
        )
        engine = ASREngine(asr_config)
        # start == end -> 0 samples
        result = engine.transcribe_segment(audio, start=0.5, end=0.5)
        assert result is None

    @patch("meeting_intelligence.transcription.asr.WhisperModel")
    def test_confidence_from_avg_logprob(
        self, mock_whisper_cls, asr_config, sample_audio
    ):
        seg = _make_mock_segment(0.0, 1.0, " Test", avg_logprob=-0.15)
        mock_info = _make_mock_info()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], mock_info)
        mock_whisper_cls.return_value = mock_model

        engine = ASREngine(asr_config)
        segments, _ = engine.transcribe(sample_audio)

        assert segments[0].confidence == -0.15
