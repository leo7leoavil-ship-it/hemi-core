"""Request schemas and parameter models using Pydantic v2."""

from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class TargetScenario(str, Enum):
    """Supported technical English evaluation scenarios."""

    SYSTEM_DESIGN = "system_design_interview"
    CODE_REVIEW = "code_review"
    INCIDENT_POSTMORTEM = "incident_postmortem"
    ARCHITECTURE_PRESENTATION = "architecture_presentation"

    @classmethod
    def _missing_(cls, value: object) -> Optional["TargetScenario"]:
        """Accept case-insensitive values and the legacy (v0.x) scenario labels."""
        if not isinstance(value, str):
            return None
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        legacy = {
            "system_design": cls.SYSTEM_DESIGN,
            "code_review": cls.CODE_REVIEW,
            "thesis_defense": cls.ARCHITECTURE_PRESENTATION,
            "incident_post_mortem": cls.INCIDENT_POSTMORTEM,
            "postmortem": cls.INCIDENT_POSTMORTEM,
        }
        if key in legacy:
            return legacy[key]
        for member in cls:
            if member.value == key:
                return member
        return None


SCENARIO_DESCRIPTIONS: Dict[TargetScenario, str] = {
    TargetScenario.SYSTEM_DESIGN: (
        "A system design interview: the engineer explains scalability, data storage, "
        "caching, partitioning and trade-offs of a distributed architecture."
    ),
    TargetScenario.CODE_REVIEW: (
        "A code review: the engineer discusses readability, complexity, testing, "
        "refactoring and correctness of a change set with teammates."
    ),
    TargetScenario.INCIDENT_POSTMORTEM: (
        "An incident post-mortem: the engineer narrates timeline, root cause, impact, "
        "mitigation and follow-up actions of a production outage."
    ),
    TargetScenario.ARCHITECTURE_PRESENTATION: (
        "An architecture presentation: the engineer defends design decisions, "
        "methodology and empirical results before a technical or academic audience."
    ),
}


class TextEvaluationRequest(BaseModel):
    """Request payload for text-based evaluation."""

    candidate_text: str = Field(
        ...,
        min_length=2,
        max_length=10000,
        description="Candidate spoken response transcript or written text to evaluate in technical English",
    )
    target_scenario: TargetScenario = Field(
        default=TargetScenario.SYSTEM_DESIGN,
        description="Domain context in which candidate is participating",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "candidate_text": "We designed the database schema to normalize tables and eliminate redundant write operations, while utilizing a Redis cluster to serve high-velocity cached queries.",
                "target_scenario": "system_design_interview",
            }
        },
    )


class QueryRequest(BaseModel):
    """Direct semantic-retrieval request (RAG introspection, no voice/LLM)."""

    query_text: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="Natural language technical query to match against the corpus",
    )
    top_k: int = Field(default=3, ge=1, le=20, description="Number of chunks to retrieve")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query_text": "How does horizontal database sharding mitigate write bottleneck?",
                "top_k": 3,
            }
        }
    )


class HealthResponse(BaseModel):
    """System health check payload."""

    status: str = Field(..., description="Overall health status ('ok' | 'degraded')")
    app: str = Field(..., description="Application name")
    version: str = Field(..., description="API software version")
    chroma_ready: bool = Field(..., description="ChromaDB collection and embedder are loaded")
    asr_ready: bool = Field(..., description="faster-whisper model is loaded")
    llm_configured: bool = Field(..., description="Groq API key is configured")
    environment: str = Field(..., description="Active runtime environment")
    components: Dict[str, str] = Field(..., description="Human-readable subsystem details")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")

    model_config = ConfigDict(populate_by_name=True)
