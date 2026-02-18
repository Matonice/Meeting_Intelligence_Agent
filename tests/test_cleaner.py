"""Tests for transcript cleaning."""

from meeting_intelligence.postprocessing.cleaner import TranscriptCleaner


class TestTranscriptCleaner:
    def setup_method(self):
        self.cleaner = TranscriptCleaner()

    def test_removes_filler_words(self):
        result = self.cleaner.remove_disfluencies("So um I think uh we should")
        # Whitespace normalization is a separate step
        result = self.cleaner.normalize_whitespace(result).strip()
        assert result == "So I think we should"

    def test_removes_repeated_words(self):
        assert self.cleaner.remove_repeated_words("the the report") == "the report"

    def test_preserves_meaningful_text(self):
        text = "We need to deploy by Friday."
        assert self.cleaner.remove_disfluencies(text) == text

    def test_capitalize_sentences(self):
        assert self.cleaner.capitalize_sentences("hello. world.") == "Hello. World."

    def test_normalize_whitespace(self):
        assert self.cleaner.normalize_whitespace("too   many    spaces") == "too many spaces"

    def test_clean_transcript(self, sample_transcript):
        cleaned = self.cleaner.clean(sample_transcript)
        assert len(cleaned.segments) > 0
        for seg in cleaned.segments:
            assert seg.text.strip() == seg.text
