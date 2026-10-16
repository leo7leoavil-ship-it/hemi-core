"""End-to-end evaluation pipelines shared by the REST and WebSocket transports.

voice : audio -> faster-whisper (thread) -> ChromaDB (thread) -> Groq LLM -> EvaluationResponse
text  : text  ->                            ChromaDB (thread) -> Groq LLM -> EvaluationResponse

Transport layers translate :class:`PipelineError` into HTTP status codes / WS error events.
"""

import time
from typing import Awaitable, Callable, Dict, Optional

from app.core.logging import logger
from app.schemas.request_schemas import TargetScenario
from app.schemas.response_schemas import EvaluationResponse
from app.services.asr_service import SILENCE_PLACEHOLDER, ASRService
from app.services.llm_service import LLMService, LLMServiceError
from app.services.rag_service import RAGService

EventCallback = Callable[[Dict], Awaitable[None]]

RAG_TOP_K = 3


class PipelineError(Exception):
    """Business error carrying the HTTP status code that best describes it."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


async def _emit(callback: Optional[EventCallback], event: Dict) -> None:
    if callback is not None:
        await callback(event)


async def _evaluate(
    text: str,
    scenario: TargetScenario,
    asr_metrics,
    rag: RAGService,
    llm: LLMService,
    start_time: float,
    on_event: Optional[EventCallback],
) -> EvaluationResponse:
    await _emit(on_event, {"event": "retrieving", "partial_text": text})
    rag_context = await rag.aquery_corpus(text, top_k=RAG_TOP_K, scenario=scenario.value)

    await _emit(on_event, {"event": "evaluating"})
    try:
        return await llm.evaluate_technical_english(
            candidate_text=text,
            target_scenario=scenario.value,
            rag_context_items=rag_context,
            asr_metrics=asr_metrics,
            start_time_monotonic=start_time,
        )
    except LLMServiceError as e:
        logger.error(f"LLM reasoning failure: {e}")
        raise PipelineError(502, f"LLM reasoning engine failure: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected LLM error: {e}", exc_info=True)
        raise PipelineError(502, f"LLM reasoning engine failure: {e}") from e


async def run_voice_pipeline(
    audio_bytes: bytes,
    scenario: TargetScenario,
    asr: ASRService,
    rag: RAGService,
    llm: LLMService,
    expected_transcript: Optional[str] = None,
    on_event: Optional[EventCallback] = None,
) -> EvaluationResponse:
    """Run the full multimodal pipeline on a complete audio payload."""
    start_time = time.perf_counter()

    if not audio_bytes:
        raise PipelineError(400, "Audio payload is empty.")

    await _emit(on_event, {"event": "transcribing"})
    try:
        asr_metrics = await asr.atranscribe_audio_stream(audio_bytes, expected_transcript)
    except ValueError as e:
        raise PipelineError(422, f"Audio decoding error: {e}") from e
    except RuntimeError as e:
        raise PipelineError(503, f"ASR engine unavailable: {e}") from e
    except Exception as e:
        logger.error(f"Whisper ASR inference error: {e}", exc_info=True)
        raise PipelineError(500, f"ASR engine failure: {e}") from e

    transcript = asr_metrics.transcription.strip()
    if not transcript or transcript == SILENCE_PLACEHOLDER:
        raise PipelineError(422, "No intelligible speech was detected in the audio signal.")

    return await _evaluate(transcript, scenario, asr_metrics, rag, llm, start_time, on_event)


async def run_text_pipeline(
    text: str,
    scenario: TargetScenario,
    rag: RAGService,
    llm: LLMService,
    on_event: Optional[EventCallback] = None,
) -> EvaluationResponse:
    """Run the text-only pipeline (RAG -> LLM)."""
    start_time = time.perf_counter()
    clean_text = text.strip()
    if len(clean_text) < 2:
        raise PipelineError(422, "candidate_text must be at least 2 characters long.")
    return await _evaluate(clean_text, scenario, None, rag, llm, start_time, on_event)
