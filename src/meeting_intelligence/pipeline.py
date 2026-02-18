"""Main pipeline orchestrator for the meeting intelligence system."""

from __future__ import annotations

import time
from pathlib import Path

from .audio.loader import load_audio
from .audio.preprocessor import VADProcessor
from .config import AppConfig
from .diarization.diarizer import SpeakerDiarizer
from .models.meeting import MeetingResult, MeetingSummary
from .models.transcript import Transcript
from .output.json_writer import JSONWriter
from .output.markdown_writer import MarkdownWriter
from .postprocessing.cleaner import TranscriptCleaner
from .postprocessing.segmenter import TopicSegmenter
from .transcription.assembler import TranscriptAssembler
from .transcription.asr import ASREngine
from .tts.speaker import SummaryNarrator
from .understanding.extractor import MeetingExtractor
from .understanding.llm_client import LLMClient
from .understanding.qa import MeetingQA
from .understanding.summarizer import MeetingSummarizer
from .utils.logging import get_logger
from .utils.progress import ProgressCallback, RichProgressCallback

logger = get_logger(__name__)


class Pipeline:
    """Orchestrates the full meeting intelligence pipeline.

    Components are lazily initialized on first use to avoid loading
    heavy models (Whisper, pyannote) when not needed.
    """

    def __init__(
        self,
        config: AppConfig,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.config = config
        self.progress = progress or RichProgressCallback()

        # Lazy-initialized components
        self._vad: VADProcessor | None = None
        self._diarizer: SpeakerDiarizer | None = None
        self._asr: ASREngine | None = None
        self._llm: LLMClient | None = None
        self._assembler = TranscriptAssembler()
        self._cleaner = TranscriptCleaner()

    @property
    def vad(self) -> VADProcessor:
        if self._vad is None:
            self._vad = VADProcessor(self.config.vad)
        return self._vad

    @property
    def diarizer(self) -> SpeakerDiarizer:
        if self._diarizer is None:
            self._diarizer = SpeakerDiarizer(self.config.diarization, device=self.config.pipeline.device)
        return self._diarizer

    @property
    def asr(self) -> ASREngine:
        if self._asr is None:
            self._asr = ASREngine(self.config.asr, device=self.config.pipeline.device)
        return self._asr

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            if not self.config.llm.api_key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY env var or configure in config.yaml"
                )
            self._llm = LLMClient(self.config.llm)
        return self._llm

    def _run_stage(self, stage: str, description: str, func, *args, **kwargs):
        """Run a pipeline stage with progress reporting and timing."""
        self.progress.on_stage_start(stage, description)
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            self.progress.on_stage_complete(stage, duration)
            return result
        except Exception as e:
            duration = time.perf_counter() - start
            self.progress.on_stage_error(stage, e)
            raise

    def process(
        self,
        audio_path: Path,
        num_speakers: int | None = None,
        language: str | None = None,
        skip_llm: bool = False,
        enable_tts: bool = False,
        questions: list[str] | None = None,
    ) -> MeetingResult:
        """Run the full pipeline on an audio file.

        Args:
            audio_path: Path to the audio file.
            num_speakers: Expected number of speakers (optional).
            language: Language code, or None for auto-detect.
            skip_llm: If True, skip GPT-4 analysis (transcript only output).
            enable_tts: If True, generate audio summary.
            questions: Optional list of questions to answer about the meeting.

        Returns:
            Complete MeetingResult with transcript, summary, decisions, etc.
        """
        audio_path = Path(audio_path)
        logger.info(f"Processing: {audio_path.name}")

        # 1. Load audio
        audio = self._run_stage(
            "load", "Loading audio file...",
            load_audio, audio_path, self.config.audio.target_sample_rate,
        )

        # 2. Voice Activity Detection
        speech_segments = self._run_stage(
            "vad", "Detecting speech regions...",
            self.vad.detect_speech, audio,
        )

        # 3. Speaker Diarization
        diarization_segments = self._run_stage(
            "diarization", "Identifying speakers...",
            self.diarizer.diarize, audio, num_speakers,
        )

        # 4. ASR Transcription
        asr_segments, detected_lang = self._run_stage(
            "asr", "Transcribing speech...",
            self.asr.transcribe, audio, language or self.config.pipeline.language,
        )

        # 5. Assemble transcript
        transcript = self._run_stage(
            "assembly", "Assembling speaker-attributed transcript...",
            self._assembler.assemble,
            asr_segments, diarization_segments,
            audio.duration_seconds, detected_lang,
        )

        # 6. Clean transcript
        transcript = self._run_stage(
            "cleaning", "Cleaning transcript...",
            self._cleaner.clean, transcript,
        )

        if skip_llm:
            # Return minimal result without LLM analysis
            summary = MeetingSummary(
                title=audio_path.stem,
                executive_summary="LLM analysis skipped.",
                key_points=[],
                topics_discussed=[],
                participant_count=len(transcript.speakers),
                duration_minutes=round(audio.duration_seconds / 60.0, 1),
            )
            return MeetingResult(
                audio_file=str(audio_path),
                transcript=transcript,
                summary=summary,
                decisions=[],
                action_items=[],
            )

        # 7. Topic Segmentation
        segmenter = TopicSegmenter(self.llm)
        topic_blocks = self._run_stage(
            "topics", "Identifying topics...",
            segmenter.segment, transcript,
        )
        transcript = transcript.model_copy(update={"topic_blocks": topic_blocks})

        # 8. Extract decisions and action items
        extractor = MeetingExtractor(self.llm)
        decisions = self._run_stage(
            "decisions", "Extracting decisions...",
            extractor.extract_decisions, transcript,
        )
        action_items = self._run_stage(
            "actions", "Extracting action items...",
            extractor.extract_action_items, transcript,
        )

        # 9. Generate summary
        summarizer = MeetingSummarizer(self.llm)
        summary = self._run_stage(
            "summary", "Generating summary...",
            summarizer.summarize, transcript, decisions, action_items,
        )

        result = MeetingResult(
            audio_file=str(audio_path),
            transcript=transcript,
            summary=summary,
            decisions=decisions,
            action_items=action_items,
        )

        # 10. Answer questions
        if questions:
            qa = MeetingQA(self.llm)
            qa.load_transcript(transcript)
            for q in questions:
                response = self._run_stage(
                    "qa", f"Answering: {q}",
                    qa.ask, q,
                )
                result.qa_history.append(response)

        # 11. Write output
        output_dir = Path(self.config.output.directory)
        if "json" in self.config.output.formats:
            JSONWriter().write(result, output_dir)
        if "markdown" in self.config.output.formats:
            MarkdownWriter().write(result, output_dir)

        # 12. TTS (optional)
        if enable_tts and self.config.tts.enabled or enable_tts:
            narrator = SummaryNarrator(self.config.tts)
            tts_path = output_dir / f"summary_{result.processed_at.strftime('%Y%m%d_%H%M%S')}.mp3"
            self._run_stage(
                "tts", "Generating audio summary...",
                narrator.narrate_sync, summary, tts_path,
            )

        logger.info("Pipeline complete!")
        return result

    def process_transcript_only(
        self,
        audio_path: Path,
        num_speakers: int | None = None,
        language: str | None = None,
    ) -> Transcript:
        """Run just the audio-to-transcript pipeline (no LLM)."""
        result = self.process(
            audio_path,
            num_speakers=num_speakers,
            language=language,
            skip_llm=True,
        )
        return result.transcript

    def ask_question(self, transcript: Transcript, question: str) -> str:
        """Ask a question about an existing transcript."""
        qa = MeetingQA(self.llm)
        response = qa.ask(question, transcript)
        return response.answer
