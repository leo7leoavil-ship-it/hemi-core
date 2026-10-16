"""Response data schemas using Pydantic v2.

Defines the structured output schemas for ASR transcription, RAG-grounded LLM
technical evaluations, lexical analysis, and error correction.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class FeedbackDetail(BaseModel):
    """Granular feedback item identifying either a vocabulary upgrade or a grammar issue."""

    type: Literal[
        "vocabulary_enhancement", "grammar_correction", "architectural_alignment"
    ] = Field(
        ...,
        description="Category of the linguistic feedback item",
    )
    original_phrase: str = Field(
        ...,
        description="Verbatim candidate phrase extracted from input",
    )
    suggested_phrase: str = Field(
        ...,
        description="Contextually precise technical or idiomatic replacement",
    )
    rag_source_context: str = Field(
        ...,
        description="Supporting documentation, RFC excerpt, or canonical standard reference",
    )
    explanation: str = Field(
        ...,
        description="Pedagogical explanation linking grammar rules or domain-specific nuances",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "type": "vocabulary_enhancement",
                "original_phrase": "We can put a cache to make it fast",
                "suggested_phrase": "We can introduce a distributed in-memory cache to mitigate latency",
                "rag_source_context": "System Architecture Patterns: In-memory caching layers reduce read latencies on high-throughput workloads.",
                "explanation": "Replacing informal verbs with precise systems architecture terminology enhances engineering credibility.",
            }
        },
    )


# Backward/forward-compatible alias used by the technical specification.
FeedbackItem = FeedbackDetail


class EvaluationCore(BaseModel):
    """Core quantitative metrics evaluating technical proficiency."""

    overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Holistic technical proficiency score on a 0-100 scale",
    )
    cefr_level: str = Field(
        default="B1",
        description="CEFR/MCER level estimated for the response (B1, B2, C1, C2)",
    )
    lexical_density_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of specialized engineering lexicon to total content words (0.0 - 1.0)",
    )
    grammar_accuracy_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Grammar correctness and syntactic precision score (0-100)",
    )
    technical_precision_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Conceptual and terminological precision against the RAG ground truth (0-100)",
    )
    detected_technical_terms: List[str] = Field(
        default_factory=list,
        description="List of domain-specific computer science/engineering terms identified",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "overall_score": 88.5,
                "cefr_level": "C1",
                "lexical_density_ratio": 0.42,
                "grammar_accuracy_score": 92.0,
                "technical_precision_score": 89.0,
                "detected_technical_terms": [
                    "distributed cache",
                    "latency",
                    "event-driven architecture",
                    "idempotency",
                ],
            }
        },
    )


class ASRMetrics(BaseModel):
    """Telemetry and transcript captured by the faster-whisper speech recognition engine."""

    transcription: str = Field(
        ...,
        description="Decoded speech-to-text transcript from candidate audio input",
    )
    detected_language: str = Field(
        ...,
        description="ISO language code detected by Whisper (e.g., 'en')",
    )
    speech_duration_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total duration of speech detected in seconds",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Whisper model transcription confidence score (0.0 - 1.0)",
    )
    word_count: int = Field(default=0, ge=0, description="Number of words in the transcript")
    words_per_minute: float = Field(
        default=0.0, ge=0.0, description="Speech rate (words per minute of detected speech)"
    )
    wer_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Word Error Rate vs expected_transcript (only when a reference is supplied)",
    )
    cer_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Character Error Rate vs expected_transcript (only when a reference is supplied)",
    )
    avg_logprob: Optional[float] = Field(
        default=None, description="Mean Whisper segment log-probability"
    )
    no_speech_prob: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Mean probability that segments contain no speech"
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "transcription": "In our microservices architecture, we implemented eventual consistency.",
                "detected_language": "en",
                "speech_duration_seconds": 4.85,
                "confidence_score": 0.96,
                "word_count": 9,
                "words_per_minute": 111.3,
                "wer_score": None,
            }
        },
    )


class EvaluationResponse(BaseModel):
    """Comprehensive evaluation payload returned by Tech Tutor Engine endpoints."""

    status: str = Field(
        default="success",
        description="Status of evaluation pipeline execution ('success' | 'partial' | 'error')",
    )
    execution_time_ms: float = Field(
        ...,
        ge=0.0,
        description="End-to-end pipeline execution time in milliseconds",
    )
    asr_metrics: Optional[ASRMetrics] = Field(
        default=None,
        description="Speech-to-text metrics if request originated from audio input",
    )
    evaluation_core: EvaluationCore = Field(
        ...,
        description="Quantitative scores and detected technical vocabulary",
    )
    feedback_details: List[FeedbackDetail] = Field(
        default_factory=list,
        description="Actionable corrective and enrichment items grounded in RAG corpus",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "status": "success",
                "execution_time_ms": 782.4,
                "asr_metrics": None,
                "evaluation_core": {
                    "overall_score": 85.0,
                    "cefr_level": "B2",
                    "lexical_density_ratio": 0.38,
                    "grammar_accuracy_score": 90.0,
                    "technical_precision_score": 84.0,
                    "detected_technical_terms": ["microservices", "eventual consistency"],
                },
                "feedback_details": [
                    {
                        "type": "vocabulary_enhancement",
                        "original_phrase": "We make the services talk together",
                        "suggested_phrase": "We enable inter-service communication through asynchronous message brokers",
                        "rag_source_context": "Distributed Systems: Event streams decouple microservices.",
                        "explanation": "Elevates informal phrasing to formal software engineering terminology.",
                    }
                ],
            }
        },
    )


class RetrievedChunk(BaseModel):
    """One semantic-search hit returned by POST /api/v1/query."""

    document: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    distance: float = Field(..., description="Cosine distance (0 = identical)")
    source: str


class QueryResponse(BaseModel):
    """Response payload of the RAG introspection endpoint."""

    status: str = "success"
    query_text: str
    top_k: int
    chunks: List[RetrievedChunk]
