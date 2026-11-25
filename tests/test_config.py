"""Unit tests for Settings configuration."""

import pytest

from app.core.config import Settings, get_settings


def fresh(**kwargs) -> Settings:
    """Settings that ignore the developer's real .env file."""
    return Settings(_env_file=None, **kwargs)


def test_defaults_match_specification(monkeypatch):
    for var in ("API_AUTH_TOKEN", "API_TOKEN", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    settings = fresh()
    assert settings.PROJECT_NAME == "hemi-core"
    assert settings.LLM_MODEL == "openai/gpt-oss-120b"
    assert settings.LLM_TEMPERATURE == 0.1
    assert settings.GROQ_BASE_URL == "https://api.groq.com/openai/v1"
    assert settings.CHROMA_PERSIST_DIRECTORY == "./data/technical_corpus_db"
    assert settings.CHROMA_COLLECTION_NAME == "hemi_core_corpus"
    assert settings.EMBEDDING_MODEL_NAME == "all-MiniLM-L6-v2"
    assert settings.ASR_MODEL_NAME == "large-v3-turbo"
    assert settings.ASR_DEVICE == "cpu"
    assert settings.ASR_COMPUTE_TYPE == "int8"
    assert "PyTorch" in settings.ASR_INITIAL_PROMPT
    assert settings.MAX_AUDIO_BYTES == 5 * 1024 * 1024
    assert settings.MAX_AUDIO_SECONDS == 60.0
    assert settings.ASR_MAX_CONCURRENCY == 1
    assert settings.ASR_BEAM_SIZE == 1
    assert settings.ASR_QUEUE_TIMEOUT_SECONDS == 10.0
    assert settings.ASR_TIMEOUT_SECONDS == 90.0
    assert settings.RAG_TIMEOUT_SECONDS == 15.0
    assert settings.LLM_TIMEOUT_SECONDS == 30.0


def test_no_hardcoded_default_token(monkeypatch):
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("API_TOKEN", raising=False)
    assert fresh().API_AUTH_TOKEN == ""


@pytest.mark.parametrize(
    "env_name,attr,value",
    [
        ("API_TOKEN", "API_AUTH_TOKEN", "secret-a"),
        ("API_AUTH_TOKEN", "API_AUTH_TOKEN", "secret-b"),
        ("LLM_MODEL_NAME", "LLM_MODEL", "other/model"),
        ("LLM_BASE_URL", "GROQ_BASE_URL", "https://example.test/v1"),
        ("WHISPER_MODEL_NAME", "ASR_MODEL_NAME", "small"),
        ("WHISPER_DEVICE", "ASR_DEVICE", "cuda"),
        ("WHISPER_COMPUTE_TYPE", "ASR_COMPUTE_TYPE", "float16"),
        ("CHROMA_PERSIST_DIR", "CHROMA_PERSIST_DIRECTORY", "/tmp/chroma"),
    ],
)
def test_spec_aliases_are_accepted(monkeypatch, env_name, attr, value):
    for var in ("API_AUTH_TOKEN", "API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(env_name, value)
    assert getattr(fresh(), attr) == value


def test_cors_origins_parsing():
    settings = fresh(ALLOWED_ORIGINS="http://localhost:3000,http://localhost:8000")
    assert "http://localhost:3000" in settings.ALLOWED_ORIGINS
    assert "http://localhost:8000" in settings.ALLOWED_ORIGINS


def test_cors_origins_json_list():
    assert fresh(ALLOWED_ORIGINS='["https://a.test"]').ALLOWED_ORIGINS == ["https://a.test"]


def test_test_environment_is_isolated_from_real_secrets():
    # conftest.py forces these values before the app is imported.
    assert get_settings().API_AUTH_TOKEN == "test-token"
