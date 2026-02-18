"""Tests for the pipeline orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from meeting_intelligence.config import AppConfig, LLMConfig
from meeting_intelligence.models.audio import AudioData, AudioMetadata, SpeechSegment
from meeting_intelligence.models.meeting import (
    ActionItem,
    Decision,
    MeetingResult,
    MeetingSummary,
    QAResponse,
)
from meeting_intelligence.models.transcript import (
    DiarizationSegment,
    Transcript,
    TranscriptSegment,
    TopicBlock,
)
from meeting_intelligence.pipeline import Pipeline


@pytest.fixture
def mock_audio():
    return AudioData(
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        duration_seconds=1.0,
        metadata=AudioMetadata(
            file_path=Path("/tmp/test.wav"),
            duration_seconds=1.0,
            sample_rate=16000,
            channels=1,
            format="wav",
            file_size_bytes=32000,
        ),
    )


@pytest.fixture
def mock_transcript():
    return Transcript(
        segments=[
            TranscriptSegment(speaker="A", start=0, end=1, text="Hello world"),
        ],
        speakers=["A"],
        duration_seconds=1.0,
    )


@pytest.fixture
def mock_summary():
    return MeetingSummary(
        title="Test",
        executive_summary="Test meeting.",
        key_points=["Point 1"],
        topics_discussed=["Topic 1"],
        participant_count=1,
        duration_minutes=0.0,
    )


@pytest.fixture
def pipeline_config():
    config = AppConfig()
    config.llm.api_key = "test-key"
    config.output.directory = "/tmp/test_output"
    return config


@pytest.fixture
def mock_progress():
    return MagicMock()


class TestPipelineLazyProperties:
    def test_llm_raises_without_api_key(self):
        config = AppConfig()
        config.llm.api_key = ""
        pipeline = Pipeline(config)
        with pytest.raises(ValueError, match="OpenAI API key required"):
            _ = pipeline.llm

    def test_vad_lazy_creation(self, pipeline_config):
        pipeline = Pipeline(pipeline_config)
        assert pipeline._vad is None
        _ = pipeline.vad
        assert pipeline._vad is not None

    def test_diarizer_lazy_creation(self, pipeline_config):
        pipeline = Pipeline(pipeline_config)
        assert pipeline._diarizer is None

    def test_asr_lazy_creation(self, pipeline_config):
        pipeline = Pipeline(pipeline_config)
        assert pipeline._asr is None


class TestPipelineRunStage:
    def test_run_stage_success(self, pipeline_config, mock_progress):
        pipeline = Pipeline(pipeline_config, progress=mock_progress)

        result = pipeline._run_stage("test", "Testing...", lambda: 42)

        assert result == 42
        mock_progress.on_stage_start.assert_called_once_with("test", "Testing...")
        mock_progress.on_stage_complete.assert_called_once()

    def test_run_stage_error(self, pipeline_config, mock_progress):
        pipeline = Pipeline(pipeline_config, progress=mock_progress)

        with pytest.raises(RuntimeError, match="boom"):
            pipeline._run_stage("test", "Testing...", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

        mock_progress.on_stage_error.assert_called_once()

    def test_run_stage_passes_args(self, pipeline_config, mock_progress):
        pipeline = Pipeline(pipeline_config, progress=mock_progress)

        result = pipeline._run_stage("test", "Testing...", lambda x, y: x + y, 3, 4)

        assert result == 7


class TestPipelineProcess:
    @patch("meeting_intelligence.pipeline.JSONWriter")
    @patch("meeting_intelligence.pipeline.MarkdownWriter")
    @patch("meeting_intelligence.pipeline.MeetingSummarizer")
    @patch("meeting_intelligence.pipeline.MeetingExtractor")
    @patch("meeting_intelligence.pipeline.TopicSegmenter")
    @patch("meeting_intelligence.pipeline.load_audio")
    def test_skip_llm_mode(
        self,
        mock_load_audio,
        mock_segmenter_cls,
        mock_extractor_cls,
        mock_summarizer_cls,
        mock_md_writer_cls,
        mock_json_writer_cls,
        pipeline_config,
        mock_progress,
        mock_audio,
        mock_transcript,
    ):
        mock_load_audio.return_value = mock_audio

        pipeline = Pipeline(pipeline_config, progress=mock_progress)

        # Mock the lazy properties
        mock_vad = MagicMock()
        mock_vad.detect_speech.return_value = [SpeechSegment(start=0, end=1)]
        pipeline._vad = mock_vad

        mock_diarizer = MagicMock()
        mock_diarizer.diarize.return_value = [
            DiarizationSegment(speaker="A", start=0, end=1)
        ]
        pipeline._diarizer = mock_diarizer

        mock_asr = MagicMock()
        mock_asr.transcribe.return_value = (
            [TranscriptSegment(speaker="", start=0, end=1, text="Hello")],
            "en",
        )
        pipeline._asr = mock_asr

        # Mock assembler and cleaner
        pipeline._assembler = MagicMock()
        pipeline._assembler.assemble.return_value = mock_transcript
        pipeline._cleaner = MagicMock()
        pipeline._cleaner.clean.return_value = mock_transcript

        # Create dummy file
        audio_path = Path("/tmp/test.wav")

        result = pipeline.process(audio_path, skip_llm=True)

        assert isinstance(result, MeetingResult)
        assert result.decisions == []
        assert result.action_items == []
        assert result.summary.executive_summary == "LLM analysis skipped."

        # LLM modules should NOT be called
        mock_segmenter_cls.assert_not_called()
        mock_extractor_cls.assert_not_called()
        mock_summarizer_cls.assert_not_called()

    @patch("meeting_intelligence.pipeline.JSONWriter")
    @patch("meeting_intelligence.pipeline.MarkdownWriter")
    @patch("meeting_intelligence.pipeline.MeetingSummarizer")
    @patch("meeting_intelligence.pipeline.MeetingExtractor")
    @patch("meeting_intelligence.pipeline.TopicSegmenter")
    @patch("meeting_intelligence.pipeline.LLMClient")
    @patch("meeting_intelligence.pipeline.load_audio")
    def test_full_pipeline(
        self,
        mock_load_audio,
        mock_llm_cls,
        mock_segmenter_cls,
        mock_extractor_cls,
        mock_summarizer_cls,
        mock_md_writer_cls,
        mock_json_writer_cls,
        pipeline_config,
        mock_progress,
        mock_audio,
        mock_transcript,
        mock_summary,
    ):
        mock_load_audio.return_value = mock_audio
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        pipeline = Pipeline(pipeline_config, progress=mock_progress)

        # Mock audio pipeline components
        mock_vad = MagicMock()
        mock_vad.detect_speech.return_value = [SpeechSegment(start=0, end=1)]
        pipeline._vad = mock_vad

        mock_diarizer = MagicMock()
        mock_diarizer.diarize.return_value = [
            DiarizationSegment(speaker="A", start=0, end=1)
        ]
        pipeline._diarizer = mock_diarizer

        mock_asr = MagicMock()
        mock_asr.transcribe.return_value = (
            [TranscriptSegment(speaker="", start=0, end=1, text="Hello")],
            "en",
        )
        pipeline._asr = mock_asr

        pipeline._assembler = MagicMock()
        pipeline._assembler.assemble.return_value = mock_transcript
        pipeline._cleaner = MagicMock()
        pipeline._cleaner.clean.return_value = mock_transcript

        # Mock LLM components
        mock_segmenter = MagicMock()
        mock_segmenter.segment.return_value = []
        mock_segmenter_cls.return_value = mock_segmenter

        mock_extractor = MagicMock()
        mock_extractor.extract_decisions.return_value = [
            Decision(id=1, text="Deploy", confidence=0.9)
        ]
        mock_extractor.extract_action_items.return_value = [
            ActionItem(id=1, task="Report")
        ]
        mock_extractor_cls.return_value = mock_extractor

        mock_summarizer = MagicMock()
        mock_summarizer.summarize.return_value = mock_summary
        mock_summarizer_cls.return_value = mock_summarizer

        result = pipeline.process(Path("/tmp/test.wav"))

        assert isinstance(result, MeetingResult)
        assert len(result.decisions) == 1
        assert len(result.action_items) == 1
        assert result.summary.title == "Test"

    @patch("meeting_intelligence.pipeline.JSONWriter")
    @patch("meeting_intelligence.pipeline.MarkdownWriter")
    @patch("meeting_intelligence.pipeline.MeetingSummarizer")
    @patch("meeting_intelligence.pipeline.MeetingExtractor")
    @patch("meeting_intelligence.pipeline.TopicSegmenter")
    @patch("meeting_intelligence.pipeline.MeetingQA")
    @patch("meeting_intelligence.pipeline.LLMClient")
    @patch("meeting_intelligence.pipeline.load_audio")
    def test_qa_during_processing(
        self,
        mock_load_audio,
        mock_llm_cls,
        mock_qa_cls,
        mock_segmenter_cls,
        mock_extractor_cls,
        mock_summarizer_cls,
        mock_md_writer_cls,
        mock_json_writer_cls,
        pipeline_config,
        mock_progress,
        mock_audio,
        mock_transcript,
        mock_summary,
    ):
        mock_load_audio.return_value = mock_audio

        pipeline = Pipeline(pipeline_config, progress=mock_progress)

        # Set up audio pipeline mocks
        pipeline._vad = MagicMock()
        pipeline._vad.detect_speech.return_value = []
        pipeline._diarizer = MagicMock()
        pipeline._diarizer.diarize.return_value = []
        pipeline._asr = MagicMock()
        pipeline._asr.transcribe.return_value = ([], "en")
        pipeline._assembler = MagicMock()
        pipeline._assembler.assemble.return_value = mock_transcript
        pipeline._cleaner = MagicMock()
        pipeline._cleaner.clean.return_value = mock_transcript

        # LLM mocks
        mock_segmenter_cls.return_value = MagicMock(segment=MagicMock(return_value=[]))
        mock_extractor = MagicMock()
        mock_extractor.extract_decisions.return_value = []
        mock_extractor.extract_action_items.return_value = []
        mock_extractor_cls.return_value = mock_extractor
        mock_summarizer_cls.return_value = MagicMock(
            summarize=MagicMock(return_value=mock_summary)
        )

        mock_qa = MagicMock()
        mock_qa.ask.return_value = QAResponse(
            question="Q", answer="A", confidence=0.8
        )
        mock_qa_cls.return_value = mock_qa

        result = pipeline.process(
            Path("/tmp/test.wav"),
            questions=["What happened?", "Who decided?"],
        )

        assert len(result.qa_history) == 2
        assert mock_qa.ask.call_count == 2

    @patch("meeting_intelligence.pipeline.load_audio")
    def test_process_transcript_only(
        self,
        mock_load_audio,
        pipeline_config,
        mock_audio,
        mock_transcript,
    ):
        mock_load_audio.return_value = mock_audio

        pipeline = Pipeline(pipeline_config)
        pipeline._vad = MagicMock()
        pipeline._vad.detect_speech.return_value = []
        pipeline._diarizer = MagicMock()
        pipeline._diarizer.diarize.return_value = []
        pipeline._asr = MagicMock()
        pipeline._asr.transcribe.return_value = ([], "en")
        pipeline._assembler = MagicMock()
        pipeline._assembler.assemble.return_value = mock_transcript
        pipeline._cleaner = MagicMock()
        pipeline._cleaner.clean.return_value = mock_transcript

        result = pipeline.process_transcript_only(Path("/tmp/test.wav"))

        assert isinstance(result, Transcript)
        assert result.speakers == ["A"]

    @patch("meeting_intelligence.pipeline.LLMClient")
    def test_ask_question(self, mock_llm_cls, pipeline_config, mock_transcript):
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        # Need to patch MeetingQA inside the method
        with patch("meeting_intelligence.pipeline.MeetingQA") as mock_qa_cls:
            mock_qa = MagicMock()
            mock_qa.ask.return_value = QAResponse(
                question="Q", answer="Deploy Friday", confidence=0.9
            )
            mock_qa_cls.return_value = mock_qa

            pipeline = Pipeline(pipeline_config)
            answer = pipeline.ask_question(mock_transcript, "When do we deploy?")

        assert answer == "Deploy Friday"


class TestPipelineEmptyQuestions:
    @patch("meeting_intelligence.pipeline.JSONWriter")
    @patch("meeting_intelligence.pipeline.MarkdownWriter")
    @patch("meeting_intelligence.pipeline.MeetingSummarizer")
    @patch("meeting_intelligence.pipeline.MeetingExtractor")
    @patch("meeting_intelligence.pipeline.TopicSegmenter")
    @patch("meeting_intelligence.pipeline.LLMClient")
    @patch("meeting_intelligence.pipeline.load_audio")
    def test_empty_questions_list_skips_qa(
        self,
        mock_load_audio,
        mock_llm_cls,
        mock_segmenter_cls,
        mock_extractor_cls,
        mock_summarizer_cls,
        mock_md_writer_cls,
        mock_json_writer_cls,
        pipeline_config,
        mock_audio,
        mock_transcript,
        mock_summary,
    ):
        mock_load_audio.return_value = mock_audio

        pipeline = Pipeline(pipeline_config)
        pipeline._vad = MagicMock(detect_speech=MagicMock(return_value=[]))
        pipeline._diarizer = MagicMock(diarize=MagicMock(return_value=[]))
        pipeline._asr = MagicMock(transcribe=MagicMock(return_value=([], "en")))
        pipeline._assembler = MagicMock(assemble=MagicMock(return_value=mock_transcript))
        pipeline._cleaner = MagicMock(clean=MagicMock(return_value=mock_transcript))

        mock_segmenter_cls.return_value = MagicMock(segment=MagicMock(return_value=[]))
        mock_ext = MagicMock()
        mock_ext.extract_decisions.return_value = []
        mock_ext.extract_action_items.return_value = []
        mock_extractor_cls.return_value = mock_ext
        mock_summarizer_cls.return_value = MagicMock(
            summarize=MagicMock(return_value=mock_summary)
        )

        result = pipeline.process(Path("/tmp/test.wav"), questions=[])

        assert result.qa_history == []
