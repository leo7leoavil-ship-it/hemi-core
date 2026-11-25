"""Application Configuration Module.

Loads environment variables using Pydantic Settings v2 with strict validation,
type safety, and fallback defaults aligned with the thesis requirements.

Naming convention
-----------------
Every setting exposes its canonical name and, where the technical specification
(``Especificacion_tech_engine.md``) uses a different identifier, an additional
alias. Both spellings are accepted from the environment / ``.env`` file, e.g.
``API_AUTH_TOKEN`` or ``API_TOKEN``; ``LLM_MODEL`` or ``LLM_MODEL_NAME``.
"""

import json
from functools import lru_cache
from typing import List, Union

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized production configuration for hemi-core."""

    # Project Metadata
    PROJECT_NAME: str = "hemi-core"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "AI Assistant Specialized in Technical English for Computer Engineers"
    )
    ENVIRONMENT: str = Field(default="production", description="Environment mode")
    DEBUG: bool = Field(default=False, description="Debug mode")

    # Server Bindings
    HOST: str = Field(default="0.0.0.0", description="Host address")
    PORT: int = Field(default=8000, description="Listening port (HF Spaces uses 7860)")

    # Groq Cloud LLM Settings
    GROQ_API_KEY: str = Field(
        default="",
        description="Groq Cloud API Key (secret; never commit it)",
    )
    GROQ_BASE_URL: str = Field(
        default="https://api.groq.com/openai/v1",
        validation_alias=AliasChoices("GROQ_BASE_URL", "LLM_BASE_URL"),
        description="OpenAI-compatible Groq Cloud endpoint base URL",
    )
    LLM_MODEL: str = Field(
        default="openai/gpt-oss-120b",
        validation_alias=AliasChoices("LLM_MODEL", "LLM_MODEL_NAME"),
        description="Target model identifier hosted on Groq Cloud",
    )
    LLM_TEMPERATURE: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Low temperature for deterministic, structured reasoning",
    )
    LLM_REASONING_EFFORT: str = Field(
        default="low",
        description="gpt-oss reasoning effort (low/medium/high). Lower = less latency",
    )
    LLM_STRUCTURED_OUTPUT: bool = Field(
        default=True,
        description="Request strict json_schema Structured Outputs (auto-fallback to json_object)",
    )
    LLM_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0.0)

    # API Security & Access Token (fail-closed: empty token rejects every request)
    API_AUTH_TOKEN: str = Field(
        default="",
        validation_alias=AliasChoices("API_AUTH_TOKEN", "API_TOKEN"),
        description="Bearer/X-API-Key token used for securing every endpoint",
    )

    # Vector Database (ChromaDB RAG Pipeline)
    CHROMA_PERSIST_DIRECTORY: str = Field(
        default="./data/technical_corpus_db",
        validation_alias=AliasChoices("CHROMA_PERSIST_DIRECTORY", "CHROMA_PERSIST_DIR"),
        description="Filesystem persistence directory for ChromaDB",
    )
    CHROMA_COLLECTION_NAME: str = Field(
        default="hemi_core_corpus",
        description="Target ChromaDB collection containing indexed IT corpus",
    )
    EMBEDDING_MODEL_NAME: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence Transformer embedding model name (384 dimensions)",
    )

    # Corpus provisioning (Hugging Face Hub dataset, "Enfoque B" of the spec)
    CORPUS_AUTO_DOWNLOAD: bool = Field(
        default=True,
        description="Download the vector DB from the HF dataset at startup if it is missing",
    )
    CORPUS_DATASET_REPO: str = Field(
        default="leonardo-beltran/hemi-core-corpus",
        description="HF dataset repo id hosting the persisted ChromaDB directory",
    )
    HF_TOKEN: str = Field(
        default="",
        description="Hugging Face access token (needed for private datasets)",
    )

    # Automatic Speech Recognition (ASR Pipeline via faster-whisper)
    ASR_MODEL_NAME: str = Field(
        default="large-v3-turbo",
        validation_alias=AliasChoices("ASR_MODEL_NAME", "WHISPER_MODEL_NAME"),
        description="faster-whisper model weight tag",
    )
    ASR_DEVICE: str = Field(
        default="cpu",
        validation_alias=AliasChoices("ASR_DEVICE", "WHISPER_DEVICE"),
        description="Inference device for ASR (cpu/cuda)",
    )
    ASR_COMPUTE_TYPE: str = Field(
        default="int8",
        validation_alias=AliasChoices("ASR_COMPUTE_TYPE", "WHISPER_COMPUTE_TYPE"),
        description="Quantization compute type (int8/float16/float32)",
    )
    ASR_CPU_THREADS: int = Field(default=4, ge=1, description="CTranslate2 CPU threads")
    ASR_BEAM_SIZE: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Whisper decoding beam size (1 = fast greedy decoding for CPU latency)",
    )
    ASR_INITIAL_PROMPT: str = Field(
        default=(
            "Technical software engineering discussion covering PyTorch, Kubernetes, "
            "Docker, FastAPI, horizontal sharding, microservices, latency, and system design."
        ),
        description="Lexical bias prompt that conditions Whisper towards engineering terms",
    )

    # Request & Pipeline Concurrency / Timeout Limits
    MAX_AUDIO_BYTES: int = Field(
        default=5 * 1024 * 1024,
        ge=1024,
        description="Maximum accepted audio payload (REST upload or WebSocket buffer)",
    )
    MAX_AUDIO_SECONDS: float = Field(
        default=60.0,
        gt=0.0,
        description="Maximum accepted audio duration in seconds",
    )
    ASR_MAX_CONCURRENCY: int = Field(
        default=1,
        ge=1,
        description="Maximum concurrent CPU ASR transcriptions",
    )
    ASR_QUEUE_TIMEOUT_SECONDS: float = Field(
        default=10.0,
        gt=0.0,
        description="Maximum wait time in seconds to acquire the ASR semaphore before HTTP 429",
    )
    ASR_TIMEOUT_SECONDS: float = Field(
        default=90.0,
        gt=0.0,
        description="Maximum execution time in seconds for the ASR transcription stage on CPU",
    )
    RAG_TIMEOUT_SECONDS: float = Field(
        default=15.0,
        gt=0.0,
        description="Maximum execution time in seconds for the RAG retrieval stage",
    )

    # Optional Gradio sandbox (out of the MaaS scope; disabled by default)
    ENABLE_UI: bool = Field(default=False, description="Mount the Gradio sandbox at /ui")

    # CORS Settings
    ALLOWED_ORIGINS: Union[List[str], str] = Field(
        default=["*"],
        description="Allowed CORS origin hosts",
    )

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
