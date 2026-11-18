"""Request schemas and parameter models using Pydantic v2."""

from enum import Enum
from typing import Dict, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator


CUSTOM_SCENARIO_MIN_LEN = 3
CUSTOM_SCENARIO_MAX_LEN = 120


class TargetScenario(str, Enum):
    """Supported technical English evaluation scenarios (9 presets + explicit custom mode)."""

    SYSTEM_DESIGN = "system_design_interview"
    CODE_REVIEW = "code_review"
    INCIDENT_POSTMORTEM = "incident_postmortem"
    ARCHITECTURE_PRESENTATION = "architecture_presentation"
    DAILY_STANDUP = "daily_standup"
    TECHNICAL_DEBUGGING = "technical_debugging"
    DEVOPS_CI_CD = "devops_ci_cd"
    DATABASE_OPTIMIZATION = "database_optimization"
    API_DESIGN_REVIEW = "api_design_review"
    CUSTOM = "custom"

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
            "standup": cls.DAILY_STANDUP,
            "debugging": cls.TECHNICAL_DEBUGGING,
            "devops": cls.DEVOPS_CI_CD,
            "database": cls.DATABASE_OPTIMIZATION,
            "api_design": cls.API_DESIGN_REVIEW,
            "custom_scenario": cls.CUSTOM,
        }
        if key in legacy:
            return legacy[key]
        for member in cls:
            if member.value == key:
                return member
        return None


SCENARIO_TITLES: Dict[TargetScenario, str] = {
    TargetScenario.SYSTEM_DESIGN: "System Design Interview",
    TargetScenario.CODE_REVIEW: "Code Review & Refactoring",
    TargetScenario.INCIDENT_POSTMORTEM: "Incident Post-Mortem & RCA",
    TargetScenario.ARCHITECTURE_PRESENTATION: "Architecture & Thesis Defense",
    TargetScenario.DAILY_STANDUP: "Agile Daily Standup & Sprint Planning",
    TargetScenario.TECHNICAL_DEBUGGING: "Live Technical Debugging & Troubleshooting",
    TargetScenario.DEVOPS_CI_CD: "DevOps, CI/CD Pipelines & Cloud Infrastructure",
    TargetScenario.DATABASE_OPTIMIZATION: "Database Modeling, Indexing & Query Optimization",
    TargetScenario.API_DESIGN_REVIEW: "API Contract, Versioning & Security Review",
    TargetScenario.CUSTOM: "Custom Engineering Scenario (User-Defined)",
}


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
    TargetScenario.DAILY_STANDUP: (
        "An agile daily standup or sprint planning sync: the engineer concisely communicates "
        "completed tasks, current technical blockers, dependencies and delivery estimates."
    ),
    TargetScenario.TECHNICAL_DEBUGGING: (
        "A live debugging session: the engineer formulates hypotheses, analyzes stack traces, "
        "memory leaks or race conditions, and proposes reproducible fixes."
    ),
    TargetScenario.DEVOPS_CI_CD: (
        "A DevOps and cloud infrastructure review: the engineer explains containerization, "
        "CI/CD pipelines, Kubernetes orchestration, observability and zero-downtime rollouts."
    ),
    TargetScenario.DATABASE_OPTIMIZATION: (
        "A database engineering discussion: the engineer evaluates schema normalization, "
        "indexing strategies, query execution plans, replication lag and ACID transactions."
    ),
    TargetScenario.API_DESIGN_REVIEW: (
        "An API design and security review: the engineer discusses REST/GraphQL/gRPC contracts, "
        "idempotency, pagination, rate limiting, authentication and backward compatibility."
    ),
    TargetScenario.CUSTOM: (
        "A user-defined software engineering scenario provided via custom_scenario."
    ),
}


def resolve_scenario(
    target_scenario: Union[TargetScenario, str],
    custom_scenario: Optional[str] = None,
) -> Tuple[TargetScenario, Optional[str], str, str]:
    """Validate and resolve (target_scenario, custom_scenario) across REST and WebSocket transports.

    Returns:
        Tuple of (scenario_enum, normalized_custom_scenario, effective_rag_label, effective_llm_description).

    Raises:
        ValueError: If target_scenario is unknown, or if custom_scenario rules are violated.
    """
    try:
        scenario_enum = (
            target_scenario
            if isinstance(target_scenario, TargetScenario)
            else TargetScenario(target_scenario)
        )
    except ValueError as e:
        valid_values = [s.value for s in TargetScenario]
        raise ValueError(f"Unknown target_scenario. Valid values: {valid_values}") from e

    raw_custom = str(custom_scenario).strip() if custom_scenario is not None else ""

    if scenario_enum == TargetScenario.CUSTOM:
        if not raw_custom:
            raise ValueError(
                "custom_scenario is required when target_scenario is 'custom' "
                f"({CUSTOM_SCENARIO_MIN_LEN}-{CUSTOM_SCENARIO_MAX_LEN} printable characters)."
            )
        if not (CUSTOM_SCENARIO_MIN_LEN <= len(raw_custom) <= CUSTOM_SCENARIO_MAX_LEN):
            raise ValueError(
                f"custom_scenario must be between {CUSTOM_SCENARIO_MIN_LEN} and "
                f"{CUSTOM_SCENARIO_MAX_LEN} characters long (received {len(raw_custom)})."
            )
        if not all(ch.isprintable() and ch not in "\r\n\t" for ch in raw_custom):
            raise ValueError(
                "custom_scenario must contain only single-line printable characters without control characters."
            )
        llm_desc = (
            f"Custom software engineering scenario focused on: '{raw_custom}'. "
            "Evaluate technical precision and CEFR terminology strictly within this engineering domain."
        )
        return scenario_enum, raw_custom, raw_custom, llm_desc

    if raw_custom:
        raise ValueError(
            f"custom_scenario is only allowed when target_scenario is 'custom' "
            f"(received target_scenario='{scenario_enum.value}')."
        )

    return scenario_enum, None, scenario_enum.value, SCENARIO_DESCRIPTIONS[scenario_enum]


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
        description="Domain context in which candidate is participating (or 'custom' with custom_scenario)",
    )
    custom_scenario: Optional[str] = Field(
        default=None,
        description=(
            "Custom engineering scenario description (3-120 printable single-line characters). "
            "Required when target_scenario='custom'; must be omitted for preset scenarios."
        ),
    )

    @model_validator(mode="after")
    def _validate_scenario_pair(self) -> "TextEvaluationRequest":
        scenario_enum, cleaned_custom, _, _ = resolve_scenario(
            self.target_scenario, self.custom_scenario
        )
        self.target_scenario = scenario_enum
        self.custom_scenario = cleaned_custom
        return self

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "candidate_text": "We designed the database schema to normalize tables and eliminate redundant write operations, while utilizing a Redis cluster to serve high-velocity cached queries.",
                "target_scenario": "system_design_interview",
                "custom_scenario": None,
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
