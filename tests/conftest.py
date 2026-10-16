"""Shared pytest fixtures.

The suite never loads Whisper/SentenceTransformer weights, never opens the real ChromaDB
and never calls Groq: the three services are replaced through FastAPI dependency overrides.
"""

import os

# Must be set BEFORE importing the application (the real .env may hold production secrets).
os.environ["API_AUTH_TOKEN"] = "test-token"
os.environ["GROQ_API_KEY"] = "test-groq-key"
os.environ["CORPUS_AUTO_DOWNLOAD"] = "false"
os.environ["ENABLE_UI"] = "false"

from typing import Any, Dict, List, Optional  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.schemas.response_schemas import (  # noqa: E402
    ASRMetrics,
    EvaluationCore,
    EvaluationResponse,
    FeedbackDetail,
)
from app.services.asr_service import SILENCE_PLACEHOLDER, get_asr_service  # noqa: E402
from app.services.llm_service import LLMServiceError, get_llm_service  # noqa: E402
from app.services.rag_service import get_rag_service  # noqa: E402

AUTH_HEADERS = {"Authorization": "Bearer test-token"}


class FakeCollection:
    def count(self) -> int:
        return 213439


class FakeRAG:
    collection = FakeCollection()

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def is_ready(self) -> bool:
        return True

    async def aquery_corpus(
        self, text: str, top_k: int = 3, scenario: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        self.calls.append({"text": text, "top_k": top_k, "scenario": scenario})
        return [
            {
                "document": "Sharding is a type of database partitioning.",
                "similarity": 0.69,
                "distance": 0.31,
                "metadata": {"source": "developer-roadmap/sharding.md"},
                "source": "developer-roadmap/sharding.md",
            }
        ][:top_k]


class FakeASR:
    def __init__(self) -> None:
        self.transcription = "We shard the database across multiple nodes to reduce write latency."
        self.received_expected: Optional[str] = None

    def is_ready(self) -> bool:
        return True

    async def atranscribe_audio_stream(
        self, file_bytes: bytes, expected_transcript: Optional[str] = None
    ) -> ASRMetrics:
        self.received_expected = expected_transcript
        silent = self.transcription == SILENCE_PLACEHOLDER
        return ASRMetrics(
            transcription=self.transcription,
            detected_language="en",
            speech_duration_seconds=5.4,
            confidence_score=0.96,
            word_count=0 if silent else len(self.transcription.split()),
            words_per_minute=0.0 if silent else 150.0,
            wer_score=0.05 if expected_transcript else None,
        )


class FakeLLM:
    def __init__(self) -> None:
        self.fail = False

    def is_configured(self) -> bool:
        return True

    async def evaluate_technical_english(
        self,
        candidate_text: str,
        target_scenario: str,
        rag_context_items: List[Dict[str, Any]],
        asr_metrics: Optional[ASRMetrics] = None,
        start_time_monotonic: Optional[float] = None,
    ) -> EvaluationResponse:
        if self.fail:
            raise LLMServiceError("simulated Groq outage")
        return EvaluationResponse(
            status="success",
            execution_time_ms=12.5,
            asr_metrics=asr_metrics,
            evaluation_core=EvaluationCore(
                overall_score=91.5,
                cefr_level="C1",
                lexical_density_ratio=0.3,
                grammar_accuracy_score=94.0,
                technical_precision_score=89.0,
                detected_technical_terms=["shard", "database", "write latency"],
            ),
            feedback_details=[
                FeedbackDetail(
                    type="vocabulary_enhancement",
                    original_phrase="shard the database",
                    suggested_phrase="implement horizontal sharding",
                    rag_source_context=f"scenario={target_scenario}",
                    explanation="More precise terminology.",
                )
            ],
        )


@pytest.fixture
def fakes() -> Dict[str, Any]:
    return {"rag": FakeRAG(), "asr": FakeASR(), "llm": FakeLLM()}


@pytest.fixture
def client(fakes: Dict[str, Any]):
    app.dependency_overrides[get_rag_service] = lambda: fakes["rag"]
    app.dependency_overrides[get_asr_service] = lambda: fakes["asr"]
    app.dependency_overrides[get_llm_service] = lambda: fakes["llm"]
    yield TestClient(app)
    app.dependency_overrides.clear()
