"""Tests for transcript assembly (diarization + ASR merge)."""

from meeting_intelligence.models.transcript import (
    DiarizationSegment,
    TranscriptSegment,
    Word,
)
from meeting_intelligence.transcription.assembler import TranscriptAssembler


class TestTranscriptAssembler:
    def setup_method(self):
        self.assembler = TranscriptAssembler()

    def test_simple_two_speaker_merge(self, sample_asr_segments, sample_diarization_segments):
        """Two speakers, non-overlapping segments correctly assigned."""
        transcript = self.assembler.assemble(
            sample_asr_segments, sample_diarization_segments, duration_seconds=10.0
        )
        assert len(transcript.speakers) == 2
        assert transcript.segments[0].speaker == "SPEAKER_00"
        assert transcript.segments[1].speaker == "SPEAKER_01"
        assert transcript.segments[2].speaker == "SPEAKER_00"

    def test_empty_asr_segments(self, sample_diarization_segments):
        """Empty ASR input produces empty transcript."""
        transcript = self.assembler.assemble([], sample_diarization_segments)
        assert transcript.segments == []
        assert transcript.speakers == []

    def test_empty_diarization(self, sample_asr_segments):
        """No diarization assigns all to SPEAKER_00."""
        transcript = self.assembler.assemble(sample_asr_segments, [])
        assert all(s.speaker == "SPEAKER_00" for s in transcript.segments)

    def test_consecutive_same_speaker_merged(self):
        """Adjacent segments from same speaker are merged."""
        asr = [
            TranscriptSegment(speaker="", start=0.0, end=2.0, text="Hello"),
            TranscriptSegment(speaker="", start=2.5, end=4.0, text="World"),
        ]
        diarization = [
            DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=5.0),
        ]
        transcript = self.assembler.assemble(asr, diarization)
        assert len(transcript.segments) == 1
        assert transcript.segments[0].text == "Hello World"

    def test_overlap_computation(self):
        """Overlap calculation works correctly."""
        assert self.assembler._compute_overlap(0, 5, 3, 8) == 2.0
        assert self.assembler._compute_overlap(0, 3, 5, 8) == 0.0
        assert self.assembler._compute_overlap(0, 5, 0, 5) == 5.0
        assert self.assembler._compute_overlap(0, 5, 2, 3) == 1.0

    def test_word_level_speaker_split(self):
        """Segment spanning two speakers is split using word timestamps."""
        asr = [
            TranscriptSegment(
                speaker="",
                start=0.0,
                end=6.0,
                text="Yes let's do that No I disagree",
                words=[
                    Word(text="Yes", start=0.0, end=0.5, confidence=0.9),
                    Word(text="let's", start=0.6, end=1.0, confidence=0.9),
                    Word(text="do", start=1.1, end=1.4, confidence=0.9),
                    Word(text="that", start=1.5, end=2.0, confidence=0.9),
                    Word(text="No", start=3.0, end=3.3, confidence=0.9),
                    Word(text="I", start=3.4, end=3.5, confidence=0.9),
                    Word(text="disagree", start=3.6, end=4.5, confidence=0.9),
                ],
            ),
        ]
        diarization = [
            DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.5),
            DiarizationSegment(speaker="SPEAKER_01", start=2.5, end=6.0),
        ]
        transcript = self.assembler.assemble(asr, diarization)
        assert len(transcript.segments) == 2
        assert transcript.segments[0].speaker == "SPEAKER_00"
        assert "Yes" in transcript.segments[0].text
        assert transcript.segments[1].speaker == "SPEAKER_01"
        assert "disagree" in transcript.segments[1].text
