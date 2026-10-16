"""Unit tests for the LLM service: deterministic metrics, parsing and error handling."""

import json
import time

import pytest

from app.services.llm_service import (
    LLMService,
    LLMServiceError,
    compute_lexical_density,
    parse_llm_json,
)


def make_llm() -> LLMService:
    return LLMService()  # conftest sets GROQ_API_KEY; the client is never called


def test_lexical_density_is_deterministic_and_capped():
    text = "We shard the database across multiple nodes because latency is high"
    # 11 words (len>1), 3 unique terms (case-insensitive duplicates collapse)
    assert compute_lexical_density(["shard", "Database", "database", "latency"], text) == round(3 / 11, 3)
    assert compute_lexical_density(["a"] * 50 + ["b"], "two words") == 1.0
    assert compute_lexical_density([], "") == 0.0


def test_parse_llm_json_tolerates_markdown_fences_and_prose():
    assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_llm_json('Here you go: {"a": 2} Hope it helps') == {"a": 2}
    with pytest.raises(Exception):
        parse_llm_json("no json at all")


@pytest.mark.asyncio
async def test_evaluate_builds_response_and_computes_density_in_python(monkeypatch):
    llm = make_llm()
    payload = {
        "overall_score": 140,  # out of range -> clamped
        "cefr_level": "c1",
        "grammar_accuracy_score": 94,
        "technical_precision_score": 89,
        "detected_technical_terms": ["shard", "database", "nodes", "write latency"],
        "feedback_details": [
            {
                "type": "weird_type",  # unknown -> vocabulary_enhancement
                "original_phrase": "shard the database",
                "suggested_phrase": "implement horizontal sharding",
                "rag_source_context": "system-design-primer",
                "explanation": "Precise terminology.",
            }
        ],
    }
    captured = {}

    async def fake_complete(messages):
        captured["messages"] = messages
        return json.dumps(payload)

    monkeypatch.setattr(llm, "_complete", fake_complete)
    text = "We need to shard the database across multiple nodes because we are experiencing high write latency."
    result = await llm.evaluate_technical_english(
        candidate_text=text,
        target_scenario="System Design",  # legacy label is normalized
        rag_context_items=[{"document": "Sharding...", "source": "roadmap", "similarity": 0.7}],
        start_time_monotonic=time.perf_counter(),
    )

    core = result.evaluation_core
    assert core.overall_score == 100.0
    assert core.cefr_level == "C1"
    assert core.technical_precision_score == 89.0
    assert core.lexical_density_ratio == round(4 / 16, 3)
    assert result.feedback_details[0].type == "vocabulary_enhancement"
    user_message = captured["messages"][1]["content"]
    assert "system_design_interview" in user_message and "Sharding..." in user_message


@pytest.mark.asyncio
async def test_invalid_json_raises_llm_error(monkeypatch):
    llm = make_llm()

    async def fake_complete(messages):
        return "definitely not json"

    monkeypatch.setattr(llm, "_complete", fake_complete)
    with pytest.raises(LLMServiceError):
        await llm.evaluate_technical_english("text here", "code_review", [])


@pytest.mark.asyncio
async def test_missing_api_key_raises(monkeypatch):
    llm = make_llm()
    monkeypatch.setattr(llm, "api_key", "")
    with pytest.raises(LLMServiceError):
        await llm.evaluate_technical_english("text here", "code_review", [])
