"""Tests for configuration loading."""

from pathlib import Path

from meeting_intelligence.config import AppConfig, load_config


class TestConfig:
    def test_default_config(self):
        config = AppConfig()
        assert config.pipeline.device == "cpu"
        assert config.audio.target_sample_rate == 16000
        assert config.asr.model_size == "large-v2"
        assert config.llm.model == "gpt-4o"

    def test_load_config_without_file(self):
        config = load_config(Path("/nonexistent/config.yaml"))
        assert isinstance(config, AppConfig)

    def test_load_config_from_yaml(self, tmp_path):
        yaml_content = """
pipeline:
  device: "cuda"
asr:
  model_size: "small"
  beam_size: 3
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = load_config(config_file)
        assert config.pipeline.device == "cuda"
        assert config.asr.model_size == "small"
        assert config.asr.beam_size == 3
        # Defaults preserved
        assert config.llm.model == "gpt-4o"
