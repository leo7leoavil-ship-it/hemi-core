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

import io
import json
import wave
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status

from app.core.config import get_settings
from app.core.logging import logger
from app.core.security import verify_websocket_token
from app.schemas.request_schemas import TargetScenario
from app.services.asr_service import ASRService, get_asr_service
from app.services.evaluation_pipeline import PipelineError, run_voice_pipeline
from app.services.llm_service import LLMService, get_llm_service
from app.services.rag_service import RAGService, get_rag_service

router = APIRouter(tags=["Real-time Practice"])


def pcm16_to_wav(pcm: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw little-endian PCM16 samples in a WAV container so they can be decoded."""
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
    expected_transcript: Optional[str] = None
    audio_format = "webm"
    sample_rate = 16000

    async def send_event(event: Dict[str, Any]) -> None:
        await websocket.send_json(event)

    async def send_error(code: int, detail: str) -> None:
        await websocket.send_json({"event": "error", "status_code": code, "detail": detail})

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if message.get("bytes") is not None:
                buffer.extend(message["bytes"])
                if len(buffer) > settings.MAX_AUDIO_BYTES:
                    buffer.clear()
                    await send_error(
                        413, f"Audio buffer exceeded {settings.MAX_AUDIO_BYTES} bytes and was discarded."
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
                if "target_scenario" in command:
                    try:
                        scenario = TargetScenario(command["target_scenario"])
                    except ValueError:
                        await send_error(
                            422, f"Unknown target_scenario. Valid: {[s.value for s in TargetScenario]}"
                        )
                        continue
                expected_transcript = command.get("expected_transcript") or None
                audio_format = str(command.get("format", audio_format)).lower()
                try:
                    sample_rate = int(command.get("sample_rate", sample_rate))
                except (TypeError, ValueError):
                    await send_error(422, "sample_rate must be an integer.")
                    continue
            elif action == "reset":
                buffer.clear()
            elif action == "end":
                audio = bytes(buffer)
                buffer.clear()
                if audio_format == "pcm16" and audio:
                    audio = pcm16_to_wav(audio, sample_rate)
                try:
                    response = await run_voice_pipeline(
                        audio_bytes=audio,
                        scenario=scenario,
                        asr=asr,
                        rag=rag,
                        llm=llm,
                        expected_transcript=expected_transcript,
                        on_event=send_event,
                    )
                    await websocket.send_json(
                        {"event": "completed", "payload": response.model_dump(mode="json")}
                    )
                except PipelineError as e:
                    await send_error(e.status_code, e.detail)
            else:
                await send_error(400, f"Unknown action '{action}'. Use config, end or reset.")
    except WebSocketDisconnect:
        logger.info("WebSocket practice client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket practice session error: {e}", exc_info=True)
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass
