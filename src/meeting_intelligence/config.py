"""Configuration loading from YAML + environment variables."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


def _substitute_env_vars(value: str) -> str:
    """Replace ${VAR} patterns with environment variable values."""

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, "")

    return re.sub(r"\$\{(\w+)}", replacer, value)


def _walk_and_substitute(obj: dict | list | str) -> dict | list | str:
    """Recursively substitute env vars in a config dict."""
    if isinstance(obj, dict):
        return {k: _walk_and_substitute(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_substitute(item) for item in obj]
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    return obj


class PipelineConfig(BaseModel):
    device: str = "cpu"
    language: str | None = "en"


class AudioConfig(BaseModel):
    target_sample_rate: int = 16000
    target_channels: int = 1


class VADConfig(BaseModel):
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100


class DiarizationConfig(BaseModel):
    model: str = "pyannote/speaker-diarization-3.1"
    min_speakers: int | None = None
    max_speakers: int | None = None
    huggingface_token: str = ""


class ASRConfig(BaseModel):
    model_size: str = "large-v2"
    compute_type: str = "int8"
    beam_size: int = 5
    word_timestamps: bool = True


class LLMConfig(BaseModel):
    model: str = "gpt-4o"
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 4096


class TTSConfig(BaseModel):
    enabled: bool = False
    voice: str = "en-US-AndrewNeural"
    rate: str = "+0%"


class OutputConfig(BaseModel):
    directory: str = "./output"
    formats: list[str] = Field(default_factory=lambda: ["json", "markdown"])


class AppConfig(BaseModel):
    """Root configuration for the meeting intelligence pipeline."""

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load configuration from YAML file with env var substitution.

    Looks for .env file in current directory for secrets.
    Falls back to defaults if no config file provided.
    """
    load_dotenv()

    if config_path is None:
        # Try default locations
        for candidate in [Path("config.yaml"), Path("config.yml")]:
            if candidate.exists():
                config_path = candidate
                break

    if config_path and config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        raw = _walk_and_substitute(raw)
    else:
        raw = {}

    config = AppConfig(**raw)

    # Fill secrets from env if not in YAML
    if not config.llm.api_key:
        config.llm.api_key = os.environ.get("OPENAI_API_KEY", "")
    if not config.diarization.huggingface_token:
        config.diarization.huggingface_token = os.environ.get("HF_TOKEN", "")

    return config
