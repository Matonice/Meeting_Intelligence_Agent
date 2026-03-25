"""Merge diarization + ASR into speaker-attributed transcript."""

from __future__ import annotations

from ..models.transcript import (
    DiarizationSegment,
    Transcript,
    TranscriptSegment,
    Word,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TranscriptAssembler:
    """Merges ASR transcription with speaker diarization via temporal overlap."""

    def assemble(
        self,
        asr_segments: list[TranscriptSegment],
        diarization_segments: list[DiarizationSegment],
        duration_seconds: float = 0.0,
        language: str = "en",
    ) -> Transcript:
        """Merge ASR segments with diarization to produce speaker-labeled transcript.

        Algorithm:
        1. For each ASR segment, find overlapping diarization segment(s).
        2. If word timestamps exist and segment spans multiple speakers,
           split at speaker boundaries.
        3. Otherwise assign speaker with greatest temporal overlap.
        4. Merge consecutive segments from the same speaker.
        """
        if not asr_segments:
            return Transcript(
                segments=[], speakers=[], language=language, duration_seconds=duration_seconds
            )

        if not diarization_segments:
            # No diarization — assign all to SPEAKER_00
            for seg in asr_segments:
                seg.speaker = "SPEAKER_00"
            return Transcript(
                segments=asr_segments,
                speakers=["SPEAKER_00"],
                language=language,
                duration_seconds=duration_seconds,
            )

        # Assign speakers to ASR segments
        labeled_segments: list[TranscriptSegment] = []
        for seg in asr_segments:
            if seg.words:
                # Try word-level speaker assignment for better accuracy
                split = self._split_segment_by_speakers(seg, diarization_segments)
                labeled_segments.extend(split)
            else:
                # Fall back to segment-level assignment
                speaker = self._find_overlapping_speaker(seg.start, seg.end, diarization_segments)
                labeled_segments.append(seg.model_copy(update={"speaker": speaker}))

        # Merge consecutive same-speaker segments
        merged = self._merge_consecutive_same_speaker(labeled_segments)

        speakers = sorted(set(s.speaker for s in merged))
        logger.info(f"Assembly: {len(merged)} segments, {len(speakers)} speakers")

        return Transcript(
            segments=merged,
            speakers=speakers,
            language=language,
            duration_seconds=duration_seconds or (merged[-1].end if merged else 0.0),
        )

    def _compute_overlap(
        self, s1_start: float, s1_end: float, s2_start: float, s2_end: float
    ) -> float:
        """Compute temporal overlap between two time ranges."""
        overlap_start = max(s1_start, s2_start)
        overlap_end = min(s1_end, s2_end)
        return max(0.0, overlap_end - overlap_start)

    def _find_overlapping_speaker(
        self,
        start: float,
        end: float,
        diarization_segments: list[DiarizationSegment],
    ) -> str:
        """Find the speaker with maximum temporal overlap for a time range."""
        speaker_overlaps: dict[str, float] = {}
        for dseg in diarization_segments:
            overlap = self._compute_overlap(start, end, dseg.start, dseg.end)
            if overlap > 0:
                speaker_overlaps[dseg.speaker] = speaker_overlaps.get(dseg.speaker, 0.0) + overlap

        if not speaker_overlaps:
            return "UNKNOWN"

        return max(speaker_overlaps, key=speaker_overlaps.get)

    def _split_segment_by_speakers(
        self,
        segment: TranscriptSegment,
        diarization_segments: list[DiarizationSegment],
    ) -> list[TranscriptSegment]:
        """Split an ASR segment at speaker boundaries using word timestamps."""
        if not segment.words:
            speaker = self._find_overlapping_speaker(
                segment.start, segment.end, diarization_segments
            )
            return [segment.model_copy(update={"speaker": speaker})]

        # Assign each word to a speaker
        word_speakers: list[tuple[Word, str]] = []
        for word in segment.words:
            speaker = self._find_overlapping_speaker(word.start, word.end, diarization_segments)
            word_speakers.append((word, speaker))

        # Group consecutive words by speaker
        result: list[TranscriptSegment] = []
        current_words: list[Word] = []
        current_speaker = word_speakers[0][1]

        for word, speaker in word_speakers:
            if speaker != current_speaker and current_words:
                # Flush current group
                result.append(
                    TranscriptSegment(
                        speaker=current_speaker,
                        start=current_words[0].start,
                        end=current_words[-1].end,
                        text=" ".join(w.text for w in current_words),
                        words=current_words,
                        confidence=segment.confidence,
                    )
                )
                current_words = []
                current_speaker = speaker
            current_words.append(word)

        # Flush last group
        if current_words:
            result.append(
                TranscriptSegment(
                    speaker=current_speaker,
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    text=" ".join(w.text for w in current_words),
                    words=current_words,
                    confidence=segment.confidence,
                )
            )

        return result

    def _merge_consecutive_same_speaker(
        self,
        segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        """Merge adjacent segments from the same speaker."""
        if not segments:
            return []

        merged: list[TranscriptSegment] = [segments[0].model_copy()]
        for seg in segments[1:]:
            if seg.speaker == merged[-1].speaker:
                # Merge into previous
                last = merged[-1]
                merged[-1] = TranscriptSegment(
                    speaker=last.speaker,
                    start=last.start,
                    end=seg.end,
                    text=f"{last.text} {seg.text}",
                    words=last.words + seg.words,
                    confidence=min(last.confidence, seg.confidence),
                )
            else:
                merged.append(seg.model_copy())

        return merged
