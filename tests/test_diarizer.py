"""Tests for speaker diarization."""

from unittest.mock import MagicMock, patch

import pytest

from meeting_intelligence.config import DiarizationConfig
from meeting_intelligence.diarization.diarizer import SpeakerDiarizer
from meeting_intelligence.models.transcript import DiarizationSegment


@pytest.fixture
def diarization_config():
    return DiarizationConfig(
        model="pyannote/speaker-diarization-3.1",
        huggingface_token="test-token",
    )


def _make_mock_turn(start, end):
    turn = MagicMock()
    turn.start = start
    turn.end = end
    return turn


def _make_mock_pipeline(tracks):
    pipeline = MagicMock()
    diarization_result = MagicMock()
    diarization_result.itertracks.return_value = [
        (_make_mock_turn(start, end), None, speaker)
        for start, end, speaker in tracks
    ]
    pipeline.return_value = diarization_result
    return pipeline


class TestSpeakerDiarizer:
    @patch("meeting_intelligence.diarization.diarizer.torch")
    @patch("meeting_intelligence.diarization.diarizer.Pipeline")
    def test_diarize_two_speakers(
        self, mock_pipeline_cls, mock_torch, diarization_config, sample_audio
    ):
        mock_pipeline = _make_mock_pipeline([
            (0.0, 3.0, "SPEAKER_00"),
            (3.5, 6.0, "SPEAKER_01"),
            (6.5, 9.0, "SPEAKER_00"),
        ])
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_torch.cuda.is_available.return_value = False

        diarizer = SpeakerDiarizer(diarization_config)
        segments = diarizer.diarize(sample_audio)

        assert len(segments) == 3
        assert segments[0].speaker == "SPEAKER_00"
        assert segments[1].speaker == "SPEAKER_01"
        assert segments[2].speaker == "SPEAKER_00"
        assert segments[0].start == 0.0
        assert segments[0].end == 3.0

    @patch("meeting_intelligence.diarization.diarizer.torch")
    @patch("meeting_intelligence.diarization.diarizer.Pipeline")
    def test_diarize_empty_result(
        self, mock_pipeline_cls, mock_torch, diarization_config, sample_audio
    ):
        mock_pipeline = _make_mock_pipeline([])
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_torch.cuda.is_available.return_value = False

        diarizer = SpeakerDiarizer(diarization_config)
        segments = diarizer.diarize(sample_audio)

        assert segments == []

    @patch("meeting_intelligence.diarization.diarizer.torch")
    @patch("meeting_intelligence.diarization.diarizer.Pipeline")
    def test_pipeline_loaded_once(
        self, mock_pipeline_cls, mock_torch, diarization_config, sample_audio
    ):
        mock_pipeline = _make_mock_pipeline([])
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_torch.cuda.is_available.return_value = False

        diarizer = SpeakerDiarizer(diarization_config)
        diarizer.diarize(sample_audio)
        diarizer.diarize(sample_audio)

        mock_pipeline_cls.from_pretrained.assert_called_once()

    @patch("meeting_intelligence.diarization.diarizer.torch")
    @patch("meeting_intelligence.diarization.diarizer.Pipeline")
    def test_num_speakers_passed(
        self, mock_pipeline_cls, mock_torch, diarization_config, sample_audio
    ):
        mock_pipeline = _make_mock_pipeline([
            (0.0, 5.0, "SPEAKER_00"),
            (5.0, 10.0, "SPEAKER_01"),
        ])
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_torch.cuda.is_available.return_value = False

        diarizer = SpeakerDiarizer(diarization_config)
        diarizer.diarize(sample_audio, num_speakers=2)

        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["num_speakers"] == 2
        assert "min_speakers" not in call_kwargs
        assert "max_speakers" not in call_kwargs

    @patch("meeting_intelligence.diarization.diarizer.torch")
    @patch("meeting_intelligence.diarization.diarizer.Pipeline")
    def test_min_max_speakers_from_config(
        self, mock_pipeline_cls, mock_torch, sample_audio
    ):
        config = DiarizationConfig(
            model="pyannote/speaker-diarization-3.1",
            min_speakers=2,
            max_speakers=5,
        )
        mock_pipeline = _make_mock_pipeline([])
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_torch.cuda.is_available.return_value = False

        diarizer = SpeakerDiarizer(config)
        diarizer.diarize(sample_audio)

        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["min_speakers"] == 2
        assert call_kwargs["max_speakers"] == 5

    @patch("meeting_intelligence.diarization.diarizer.torch")
    @patch("meeting_intelligence.diarization.diarizer.Pipeline")
    def test_hf_token_passed_to_pipeline(
        self, mock_pipeline_cls, mock_torch, sample_audio
    ):
        config = DiarizationConfig(
            model="pyannote/speaker-diarization-3.1",
            huggingface_token="hf_test123",
        )
        mock_pipeline = _make_mock_pipeline([])
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_torch.cuda.is_available.return_value = False

        diarizer = SpeakerDiarizer(config)
        diarizer.diarize(sample_audio)

        mock_pipeline_cls.from_pretrained.assert_called_once_with(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="hf_test123",
        )

    @patch("meeting_intelligence.diarization.diarizer.torch")
    @patch("meeting_intelligence.diarization.diarizer.Pipeline")
    def test_empty_token_passes_none(
        self, mock_pipeline_cls, mock_torch, sample_audio
    ):
        config = DiarizationConfig(
            model="pyannote/speaker-diarization-3.1",
            huggingface_token="",
        )
        mock_pipeline = _make_mock_pipeline([])
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_torch.cuda.is_available.return_value = False

        diarizer = SpeakerDiarizer(config)
        diarizer.diarize(sample_audio)

        mock_pipeline_cls.from_pretrained.assert_called_once_with(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=None,
        )

    def test_get_speaker_stats(self):
        segments = [
            DiarizationSegment(speaker="A", start=0.0, end=3.0),
            DiarizationSegment(speaker="B", start=3.0, end=5.0),
            DiarizationSegment(speaker="A", start=5.0, end=8.0),
        ]
        diarizer = SpeakerDiarizer.__new__(SpeakerDiarizer)
        stats = diarizer.get_speaker_stats(segments)

        assert stats["A"] == pytest.approx(6.0)
        assert stats["B"] == pytest.approx(2.0)

    def test_get_speaker_stats_empty(self):
        diarizer = SpeakerDiarizer.__new__(SpeakerDiarizer)
        assert diarizer.get_speaker_stats([]) == {}
