"""End-to-end evaluation pipelines shared by the REST and WebSocket transports.

voice : audio -> faster-whisper (thread) -> ChromaDB (thread) -> Groq LLM -> EvaluationResponse
text  : text  ->                            ChromaDB (thread) -> Groq LLM -> EvaluationResponse

Transport layers translate :class:`PipelineError` into HTTP status codes / WS error events.
"""

import asyncio
import time
from typing import Awaitable, Callable, Dict, Optional, Union

import numpy as np

from app.core.config import get_settings
from app.core.logging import logger
from app.schemas.request_schemas import TargetScenario, resolve_scenario
from app.schemas.response_schemas import EvaluationResponse
from app.services.asr_service import (
    SILENCE_PLACEHOLDER,
    ASRQueueTimeoutError,
    ASRService,
    AudioDurationExceededError,
)
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
    custom_scenario: Optional[str] = None,
) -> EvaluationResponse:
    settings = get_settings()
    try:
        _, _, effective_scenario_label, _ = resolve_scenario(scenario, custom_scenario)
    except ValueError as e:
        raise PipelineError(422, str(e)) from e

    await _emit(on_event, {"event": "retrieving", "partial_text": text})
    try:
        rag_context = await asyncio.wait_for(
            rag.aquery_corpus(text, top_k=RAG_TOP_K, scenario=effective_scenario_label),
            timeout=settings.RAG_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as e:
        logger.error(f"RAG retrieval timed out after {settings.RAG_TIMEOUT_SECONDS}s")
        raise PipelineError(
            504, f"RAG retrieval timed out after {settings.RAG_TIMEOUT_SECONDS:.1f}s."
        ) from e

    await _emit(on_event, {"event": "evaluating"})
    try:
        return await asyncio.wait_for(
            llm.evaluate_technical_english(
                candidate_text=text,
                target_scenario=effective_scenario_label,
                rag_context_items=rag_context,
                asr_metrics=asr_metrics,
                start_time_monotonic=start_time,
            ),
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as e:
        logger.error(f"LLM evaluation timed out after {settings.LLM_TIMEOUT_SECONDS}s")
        raise PipelineError(
            504, f"LLM evaluation timed out after {settings.LLM_TIMEOUT_SECONDS:.1f}s."
        ) from e
    except LLMServiceError as e:
        logger.error(f"LLM reasoning failure: {e}")
        raise PipelineError(502, f"LLM reasoning engine failure: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected LLM error: {e}", exc_info=True)
        raise PipelineError(502, f"LLM reasoning engine failure: {e}") from e


async def run_voice_pipeline(
    audio_bytes: Union[bytes, np.ndarray],
    scenario: TargetScenario,
    asr: ASRService,
    rag: RAGService,
    llm: LLMService,
    expected_transcript: Optional[str] = None,
    on_event: Optional[EventCallback] = None,
    custom_scenario: Optional[str] = None,
) -> EvaluationResponse:
    """Run the full multimodal pipeline on a complete audio payload or PCM float32 array."""
    start_time = time.perf_counter()
    settings = get_settings()

    try:
        resolve_scenario(scenario, custom_scenario)
    except ValueError as e:
        raise PipelineError(422, str(e)) from e

    if audio_bytes is None or len(audio_bytes) == 0:
        raise PipelineError(400, "Audio payload is empty.")

    await _emit(on_event, {"event": "transcribing"})
    try:
        asr_metrics = await asyncio.wait_for(
            asr.atranscribe_audio_stream(audio_bytes, expected_transcript),
            timeout=settings.ASR_TIMEOUT_SECONDS + settings.ASR_QUEUE_TIMEOUT_SECONDS,
        )
    except ASRQueueTimeoutError as e:
        logger.warning(f"ASR queue saturation: {e}")
        raise PipelineError(429, str(e)) from e
    except asyncio.TimeoutError as e:
        logger.error(f"ASR transcription timed out after {settings.ASR_TIMEOUT_SECONDS}s")
        raise PipelineError(
            504, f"ASR transcription timed out after {settings.ASR_TIMEOUT_SECONDS:.1f}s."
        ) from e
    except AudioDurationExceededError as e:
        raise PipelineError(422, str(e)) from e
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

    return await _evaluate(
        transcript,
        scenario,
        asr_metrics,
        rag,
        llm,
        start_time,
        on_event,
        custom_scenario=custom_scenario,
    )


async def run_text_pipeline(
    text: str,
    scenario: TargetScenario,
    rag: RAGService,
    llm: LLMService,
    on_event: Optional[EventCallback] = None,
    custom_scenario: Optional[str] = None,
) -> EvaluationResponse:
    """Run the text-only pipeline (RAG -> LLM)."""
    start_time = time.perf_counter()
    clean_text = text.strip()
    if len(clean_text) < 2:
        raise PipelineError(422, "candidate_text must be at least 2 characters long.")
    return await _evaluate(
        clean_text,
        scenario,
        None,
        rag,
        llm,
        start_time,
        on_event,
        custom_scenario=custom_scenario,
    )
