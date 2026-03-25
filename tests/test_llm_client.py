"""Tests for the LLM client wrapper."""

from unittest.mock import MagicMock, patch

import pytest
from meeting_intelligence.config import LLMConfig
from meeting_intelligence.understanding.llm_client import LLMClient
from pydantic import BaseModel


class SampleResponse(BaseModel):
    """Test response model for structured output tests."""

    name: str
    value: int = 0


@pytest.fixture
def llm_config():
    return LLMConfig(api_key="test-key", model="gpt-4", temperature=0.1, max_tokens=100)


@pytest.fixture
def mock_openai():
    with patch("meeting_intelligence.understanding.llm_client.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


class TestLLMClientComplete:
    def test_complete_text_returns_string(self, llm_config, mock_openai):
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello world"
        mock_openai.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        client = LLMClient(llm_config)
        result = client.complete("system", "user")

        assert result == "Hello world"

    def test_complete_text_none_content_returns_empty(self, llm_config, mock_openai):
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_openai.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        client = LLMClient(llm_config)
        result = client.complete("system", "user")

        assert result == ""

    def test_complete_structured_native(self, llm_config, mock_openai):
        """Test native structured output path (gpt-4o+)."""
        parsed = SampleResponse(name="test", value=42)
        mock_choice = MagicMock()
        mock_choice.message.parsed = parsed
        mock_openai.beta.chat.completions.parse.return_value = MagicMock(choices=[mock_choice])

        client = LLMClient(llm_config)
        result = client.complete("system", "user", response_model=SampleResponse)

        assert isinstance(result, SampleResponse)
        assert result.name == "test"
        assert result.value == 42

    def test_complete_structured_fallback_on_exception(self, llm_config, mock_openai):
        """Falls back to JSON mode when native parse fails."""
        mock_openai.beta.chat.completions.parse.side_effect = Exception("Not supported")

        mock_choice = MagicMock()
        mock_choice.message.content = '{"name": "fallback", "value": 99}'
        mock_openai.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        client = LLMClient(llm_config)
        result = client.complete("system", "user", response_model=SampleResponse)

        assert isinstance(result, SampleResponse)
        assert result.name == "fallback"
        assert result.value == 99

    def test_complete_structured_fallback_on_none_parsed(self, llm_config, mock_openai):
        """Falls back when native parse returns None."""
        mock_choice = MagicMock()
        mock_choice.message.parsed = None
        mock_openai.beta.chat.completions.parse.return_value = MagicMock(choices=[mock_choice])

        mock_fallback_choice = MagicMock()
        mock_fallback_choice.message.content = '{"name": "fallback", "value": 1}'
        mock_openai.chat.completions.create.return_value = MagicMock(choices=[mock_fallback_choice])

        client = LLMClient(llm_config)
        result = client.complete("system", "user", response_model=SampleResponse)

        assert result.name == "fallback"

    def test_passes_temperature_and_max_tokens(self, llm_config, mock_openai):
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_openai.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        client = LLMClient(llm_config)
        client.complete("sys", "usr")

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["model"] == "gpt-4"


class TestLLMClientTokens:
    @patch("meeting_intelligence.understanding.llm_client.tiktoken")
    def test_count_tokens(self, mock_tiktoken, llm_config, mock_openai):
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [1, 2, 3, 4, 5]
        mock_tiktoken.encoding_for_model.return_value = mock_encoder

        client = LLMClient(llm_config)
        count = client.count_tokens("hello world")

        assert count == 5
        mock_encoder.encode.assert_called_once_with("hello world")

    @patch("meeting_intelligence.understanding.llm_client.tiktoken")
    def test_count_tokens_fallback_encoding(self, mock_tiktoken, llm_config, mock_openai):
        """Falls back to cl100k_base when model not in tiktoken."""
        mock_tiktoken.encoding_for_model.side_effect = KeyError("unknown model")
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [1, 2, 3]
        mock_tiktoken.get_encoding.return_value = mock_encoder

        client = LLMClient(llm_config)
        count = client.count_tokens("test")

        assert count == 3
        mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")

    @patch("meeting_intelligence.understanding.llm_client.tiktoken")
    def test_count_tokens_encoder_cached(self, mock_tiktoken, llm_config, mock_openai):
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [1, 2]
        mock_tiktoken.encoding_for_model.return_value = mock_encoder

        client = LLMClient(llm_config)
        client.count_tokens("first")
        client.count_tokens("second")

        mock_tiktoken.encoding_for_model.assert_called_once()


class TestLLMClientChunking:
    @patch("meeting_intelligence.understanding.llm_client.tiktoken")
    def test_chunk_transcript_single_chunk(self, mock_tiktoken, llm_config, mock_openai):
        mock_encoder = MagicMock()
        mock_encoder.encode.side_effect = lambda text: list(range(len(text.split())))
        mock_tiktoken.encoding_for_model.return_value = mock_encoder

        client = LLMClient(llm_config)
        chunks = client.chunk_transcript("line one\nline two\nline three", max_tokens=100)

        assert len(chunks) == 1
        assert "line one" in chunks[0]

    @patch("meeting_intelligence.understanding.llm_client.tiktoken")
    def test_chunk_transcript_splits_on_lines(self, mock_tiktoken, llm_config, mock_openai):
        mock_encoder = MagicMock()
        # Each line gets 5 tokens
        mock_encoder.encode.side_effect = lambda text: [0] * 5
        mock_tiktoken.encoding_for_model.return_value = mock_encoder

        client = LLMClient(llm_config)
        text = "line1\nline2\nline3\nline4"
        chunks = client.chunk_transcript(text, max_tokens=12)

        # 5 tokens per line, max 12: 2 lines per chunk, so 2 chunks
        assert len(chunks) == 2

    @patch("meeting_intelligence.understanding.llm_client.tiktoken")
    def test_chunk_empty_text(self, mock_tiktoken, llm_config, mock_openai):
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = []
        mock_tiktoken.encoding_for_model.return_value = mock_encoder

        client = LLMClient(llm_config)
        chunks = client.chunk_transcript("")

        assert len(chunks) == 1
        assert chunks[0] == ""
