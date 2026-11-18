"""Unit tests for Pydantic v2 response schemas."""

import pytest
from pydantic import ValidationError
from app.schemas.response_schemas import (
    ASRMetrics,
    EvaluationCore,
    EvaluationResponse,
    FeedbackDetail,
)


def test_feedback_detail_valid():
    item = FeedbackDetail(
        type="vocabulary_enhancement",
        original_phrase="We make it fast",
        suggested_phrase="We minimize latency",
        rag_source_context="Distributed Systems Caching Guide",
        explanation="Improves technical precision.",
    )
    assert item.type == "vocabulary_enhancement"
    assert item.suggested_phrase == "We minimize latency"


def test_feedback_detail_invalid_type():
    with pytest.raises(ValidationError):
        FeedbackDetail(
            type="invalid_type",  # type: ignore
            original_phrase="test",
            suggested_phrase="test",
            rag_source_context="test",
            explanation="test",
        )


def test_evaluation_core_bounds():
    core = EvaluationCore(
        overall_score=85.5,
        lexical_density_ratio=0.45,
        grammar_accuracy_score=92.0,
        detected_technical_terms=["cache", "latency", "redis"],
    )
    assert core.overall_score == 85.5
    assert len(core.detected_technical_terms) == 3

    # Invalid overall_score > 100
    with pytest.raises(ValidationError):
        EvaluationCore(
            overall_score=105.0,
            lexical_density_ratio=0.5,
            grammar_accuracy_score=90.0,
        )

    # Invalid lexical_density_ratio > 1.0
    with pytest.raises(ValidationError):
        EvaluationCore(
            overall_score=90.0,
            lexical_density_ratio=1.5,
            grammar_accuracy_score=90.0,
        )


def test_asr_metrics_model():
    metrics = ASRMetrics(
        transcription="We implemented a distributed event bus.",
        detected_language="en",
        speech_duration_seconds=3.4,
        confidence_score=0.98,
    )
    assert metrics.detected_language == "en"
    assert metrics.speech_duration_seconds == 3.4


def test_evaluation_response_serialization():
    resp = EvaluationResponse(
        status="success",
        execution_time_ms=450.2,
        asr_metrics=None,
        evaluation_core=EvaluationCore(
            overall_score=88.0,
            lexical_density_ratio=0.4,
            grammar_accuracy_score=95.0,
            detected_technical_terms=["concurrency"],
        ),
        feedback_details=[
            FeedbackDetail(
                type="grammar_correction",
                original_phrase="it do run fast",
                suggested_phrase="it executes efficiently",
                rag_source_context="Clean Code Standards",
                explanation="Subject-verb agreement and formal technical vocabulary.",
            )
        ],
    )
    dumped = resp.model_dump()
    assert dumped["status"] == "success"
    assert len(dumped["feedback_details"]) == 1
    assert dumped["execution_time_ms"] == 450.2

def test_new_contract_fields_have_safe_defaults():
    metrics = ASRMetrics(
        transcription="x", detected_language="en", speech_duration_seconds=1.0, confidence_score=0.9
    )
    assert metrics.word_count == 0 and metrics.wer_score is None
    core = EvaluationCore(overall_score=1, lexical_density_ratio=0.1, grammar_accuracy_score=1)
    assert core.cefr_level == "B1" and core.technical_precision_score == 0.0


def test_architectural_alignment_feedback_type_and_alias():
    from app.schemas.response_schemas import FeedbackItem

    item = FeedbackItem(
        type="architectural_alignment",
        original_phrase="a",
        suggested_phrase="b",
        rag_source_context="c",
        explanation="d",
    )
    assert item.type == "architectural_alignment"


def test_scenarios_and_legacy_aliases():
    from app.schemas.request_schemas import (
        SCENARIO_DESCRIPTIONS,
        SCENARIO_TITLES,
        TargetScenario,
        TextEvaluationRequest,
        resolve_scenario,
    )

    assert len(TargetScenario) == 10
    for member in TargetScenario:
        assert member in SCENARIO_DESCRIPTIONS
        assert member in SCENARIO_TITLES

    assert TargetScenario("code_review") is TargetScenario.CODE_REVIEW
    assert TargetScenario("System Design") is TargetScenario.SYSTEM_DESIGN
    assert TargetScenario("Thesis Defense") is TargetScenario.ARCHITECTURE_PRESENTATION
    assert TargetScenario("Incident-Postmortem") is TargetScenario.INCIDENT_POSTMORTEM
    assert TargetScenario("daily_standup") is TargetScenario.DAILY_STANDUP
    assert TargetScenario("technical_debugging") is TargetScenario.TECHNICAL_DEBUGGING
    assert TargetScenario("devops_ci_cd") is TargetScenario.DEVOPS_CI_CD
    assert TargetScenario("database_optimization") is TargetScenario.DATABASE_OPTIMIZATION
    assert TargetScenario("api_design_review") is TargetScenario.API_DESIGN_REVIEW
    assert TargetScenario("custom") is TargetScenario.CUSTOM
    with pytest.raises(ValueError):
        TargetScenario("cooking")

    # Custom scenario validation rules
    scen, custom, rag_label, llm_desc = resolve_scenario(
        "custom", "  Kubernetes CNI packet loss debugging  "
    )
    assert scen is TargetScenario.CUSTOM
    assert custom == "Kubernetes CNI packet loss debugging"
    assert rag_label == "Kubernetes CNI packet loss debugging"
    assert "Kubernetes CNI packet loss debugging" in llm_desc

    # Missing or too-short custom_scenario when target_scenario="custom"
    with pytest.raises(ValueError):
        resolve_scenario("custom", None)
    with pytest.raises(ValueError):
        resolve_scenario("custom", "ab")
    with pytest.raises(ValueError):
        resolve_scenario("custom", "line1\nline2")

    # Non-empty custom_scenario forbidden when target_scenario is a preset
    with pytest.raises(ValueError):
        resolve_scenario("code_review", "Some custom description")

    req = TextEvaluationRequest(
        candidate_text="We debugged the ingress controller.",
        target_scenario="custom",
        custom_scenario="Ingress controller TLS handshake debugging",
    )
    assert req.target_scenario is TargetScenario.CUSTOM
    assert req.custom_scenario == "Ingress controller TLS handshake debugging"
