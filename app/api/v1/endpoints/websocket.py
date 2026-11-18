"""Full-duplex WebSocket channel for streaming audio practice sessions (spec 4.4).

Endpoint: ``WS /api/v1/ws/practice``

Authentication
    ``Authorization: Bearer <token>`` / ``X-API-Key`` header, or ``?token=<token>`` query
    parameter (browsers cannot set headers on WebSockets).

Client -> server
    * binary frames : audio chunks (e.g. MediaRecorder ``webm`` slices, or raw PCM16).
    * text frame ``{"action": "config", "target_scenario": "code_review",
      "expected_transcript": "...", "format": "webm" | "pcm16", "sample_rate": 16000}``
      (optional; may be sent before every utterance).
    * text frame ``{"action": "end"}`` : the utterance is complete -> run the pipeline.
    * text frame ``{"action": "reset"}`` : discard the buffered audio.

Server -> client (JSON text frames)
    ``{"event": "ready"}``, ``{"event": "transcribing"}``,
    ``{"event": "retrieving", "partial_text": "..."}``, ``{"event": "evaluating"}``,
    ``{"event": "completed", "payload": {...EvaluationResponse...}}`` and
    ``{"event": "error", "status_code": int, "detail": "..."}``.
    The connection stays open after ``completed`` / ``error`` so several utterances can be
    evaluated in one session.
"""

import asyncio
import contextlib
import io
import json
import wave
from typing import Any, Dict, Optional, Union

import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status

from app.core.config import get_settings
from app.core.logging import logger
from app.core.security import verify_websocket_token
from app.schemas.request_schemas import TargetScenario, resolve_scenario
from app.services.asr_service import (
    MAX_SAMPLE_RATE,
    MIN_SAMPLE_RATE,
    ASRService,
    get_asr_service,
    pcm16_to_float32,
)
from app.services.evaluation_pipeline import PipelineError, run_voice_pipeline
from app.services.llm_service import LLMService, get_llm_service
from app.services.rag_service import RAGService, get_rag_service

router = APIRouter(tags=["Real-time Practice"])


def pcm16_to_wav(pcm: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Legacy helper wrapping raw little-endian PCM16 samples in a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


@router.websocket("/ws/practice")
async def practice_websocket(
    websocket: WebSocket,
    asr: ASRService = Depends(get_asr_service),
    rag: RAGService = Depends(get_rag_service),
    llm: LLMService = Depends(get_llm_service),
) -> None:
    settings = get_settings()

    if not verify_websocket_token(
        websocket.headers.get("authorization"),
        websocket.headers.get("x-api-key"),
        websocket.query_params.get("token"),
    ):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    await websocket.send_json({"event": "ready"})

    buffer = bytearray()
    scenario = TargetScenario.SYSTEM_DESIGN
    custom_scenario: Optional[str] = None
    expected_transcript: Optional[str] = None
    audio_format = "webm"
    sample_rate = 16000
    active_task: Optional[asyncio.Task] = None
    send_lock = asyncio.Lock()
    disconnected = False

    async def send_event(event: Dict[str, Any]) -> None:
        if disconnected:
            return
        async with send_lock:
            if not disconnected:
                await websocket.send_json(event)

    async def send_error(code: int, detail: str) -> None:
        await send_event({"event": "error", "status_code": code, "detail": detail})

    async def cancel_active_task() -> None:
        nonlocal active_task
        if active_task is not None and not active_task.done():
            active_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await active_task
        active_task = None

    async def _run_utterance(
        audio_input: Union[bytes, np.ndarray],
        utterance_scenario: TargetScenario,
        utterance_expected: Optional[str],
        utterance_custom_scenario: Optional[str],
    ) -> None:
        try:
            response = await run_voice_pipeline(
                audio_bytes=audio_input,
                scenario=utterance_scenario,
                asr=asr,
                rag=rag,
                llm=llm,
                expected_transcript=utterance_expected,
                on_event=send_event,
                custom_scenario=utterance_custom_scenario,
            )
            await send_event(
                {"event": "completed", "payload": response.model_dump(mode="json")}
            )
        except asyncio.CancelledError:
            logger.info("WebSocket voice evaluation task cancelled by client reset/disconnect.")
            raise
        except PipelineError as e:
            await send_error(e.status_code, e.detail)
        except Exception as e:
            logger.error(f"Unexpected WebSocket evaluation task failure: {e}", exc_info=True)
            await send_error(500, f"Internal evaluation pipeline error: {e}")

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                disconnected = True
                break

            if message.get("bytes") is not None:
                buffer.extend(message["bytes"])
                if len(buffer) > settings.MAX_AUDIO_BYTES:
                    buffer.clear()
                    await send_error(
                        413,
                        f"Audio buffer exceeded {settings.MAX_AUDIO_BYTES} bytes and was discarded.",
                    )
                continue

            raw_text = message.get("text")
            if raw_text is None:
                continue

            try:
                command = json.loads(raw_text)
                if not isinstance(command, dict):
                    raise ValueError("expected a JSON object")
            except ValueError:
                # Tolerate bare "end" / "reset" strings.
                command = {"action": raw_text.strip().lower()}

            action = command.get("action")

            if action == "config":
                if "target_scenario" in command or "custom_scenario" in command:
                    raw_target = command.get("target_scenario", scenario)
                    # When target_scenario is updated without custom_scenario, clear prior custom_scenario
                    raw_custom = (
                        command.get("custom_scenario")
                        if "target_scenario" in command
                        else command.get("custom_scenario", custom_scenario)
                    )
                    try:
                        scenario, custom_scenario, _, _ = resolve_scenario(raw_target, raw_custom)
                    except ValueError as e:
                        await send_error(422, str(e))
                        continue
                expected_transcript = command.get("expected_transcript") or None
                audio_format = str(command.get("format", audio_format)).lower()
                try:
                    parsed_rate = int(command.get("sample_rate", sample_rate))
                except (TypeError, ValueError):
                    await send_error(422, "sample_rate must be an integer.")
                    continue
                if not (MIN_SAMPLE_RATE <= parsed_rate <= MAX_SAMPLE_RATE):
                    await send_error(
                        422,
                        f"sample_rate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} Hz.",
                    )
                    continue
                sample_rate = parsed_rate
            elif action == "reset":
                buffer.clear()
                await cancel_active_task()
            elif action == "end":
                await cancel_active_task()
                if len(buffer) == 0:
                    await send_error(400, "Audio payload is empty.")
                    continue

                raw_buffer, buffer = buffer, bytearray()
                if audio_format == "pcm16":
                    pcm_duration = (len(raw_buffer) // 2) / float(sample_rate)
                    if pcm_duration > settings.MAX_AUDIO_SECONDS:
                        await send_error(
                            422,
                            f"Audio duration ({pcm_duration:.2f}s) exceeds maximum allowed "
                            f"duration of {settings.MAX_AUDIO_SECONDS:.1f}s.",
                        )
                        continue
                    try:
                        audio_input: Union[bytes, np.ndarray] = pcm16_to_float32(
                            raw_buffer, sample_rate
                        )
                    except ValueError as e:
                        await send_error(422, f"Audio decoding error: {e}")
                        continue
                else:
                    audio_input = bytes(raw_buffer)
                del raw_buffer

                active_task = asyncio.create_task(
                    _run_utterance(audio_input, scenario, expected_transcript, custom_scenario)
                )
            else:
                await send_error(400, f"Unknown action '{action}'. Use config, end or reset.")
    except WebSocketDisconnect:
        disconnected = True
        logger.info("WebSocket practice client disconnected.")
    except Exception as e:
        disconnected = True
        logger.error(f"WebSocket practice session error: {e}", exc_info=True)
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass
    finally:
        disconnected = True
        await cancel_active_task()
