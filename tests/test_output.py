"""Tests for output writers."""

import json

from meeting_intelligence.output.json_writer import JSONWriter
from meeting_intelligence.output.markdown_writer import MarkdownWriter


class TestJSONWriter:
    def test_write_creates_json_file(self, tmp_path, sample_meeting_result):
        writer = JSONWriter()
        path = writer.write(sample_meeting_result, tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

        data = json.loads(path.read_text())
        assert "transcript" in data
        assert "decisions" in data
        assert "action_items" in data
        assert "summary" in data

    def test_write_transcript(self, tmp_path, sample_transcript):
        writer = JSONWriter()
        out = tmp_path / "transcript.json"
        path = writer.write_transcript(sample_transcript, out)
        assert path.exists()

        data = json.loads(path.read_text())
        assert "segments" in data
        assert "speakers" in data


class TestMarkdownWriter:
    def test_write_creates_markdown(self, tmp_path, sample_meeting_result):
        writer = MarkdownWriter()
        path = writer.write(sample_meeting_result, tmp_path)
        assert path.exists()
        assert path.suffix == ".md"

        content = path.read_text()
        assert "## Executive Summary" in content
        assert "## Key Points" in content
        assert "## Decisions" in content
        assert "## Action Items" in content
        assert "## Full Transcript" in content

    def test_markdown_contains_speakers(self, tmp_path, sample_meeting_result):
        writer = MarkdownWriter()
        path = writer.write(sample_meeting_result, tmp_path)
        content = path.read_text()
        assert "SPEAKER_00" in content
        assert "SPEAKER_01" in content
