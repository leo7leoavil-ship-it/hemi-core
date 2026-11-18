# API v1 Endpoints

## Overview
HTTP REST and WebSocket route handlers for health diagnostics, direct semantic query introspection, candidate voice/text evaluation, and real-time streaming practice sessions.

## Interface
- [health.py:get_health_status]: GET `/health` endpoint returning system readiness (`HealthResponse`).
- [query.py:query_corpus_endpoint]: POST `/query` endpoint for direct semantic search on ChromaDB (`QueryRequest` -> `QueryResponse`).
- [evaluate.py:evaluate_voice_endpoint]: POST `/evaluate/voice` multipart/form-data handler executing the multimodal ASR -> RAG -> LLM pipeline (`EvaluationResponse`).
- [evaluate.py:evaluate_text_endpoint]: POST `/evaluate/text` JSON handler executing the RAG -> LLM pipeline (`EvaluationResponse`).
- [websocket.py:practice_websocket]: WebSocket `/ws/practice` interactive audio streaming session with progress telemetry events (`ready`, `transcribing`, `retrieving`, `evaluating`, `completed`, `error`).
- [websocket.py:pcm16_to_wav]: Helper packaging raw PCM16 byte buffers into WAV containers.

## Invariants
- [health.py:verify_api_token]: Every REST endpoint enforces authentication via Bearer token or `X-API-Key` header; missing or invalid tokens reject with HTTP 401.
- [websocket.py:verify_websocket_token]: WebSocket handshakes validate the token in Authorization header, `X-API-Key`, or `token` query parameter; invalid tokens close the connection with code 1008.
- [evaluate.py:VALID_AUDIO_EXTENSIONS]: Audio files must end with `.webm`, `.mp3`, `.wav`, `.ogg`, or `.m4a`.
- [evaluate.py:audio_bytes]: Empty audio payloads reject with HTTP 400; payloads exceeding `settings.MAX_AUDIO_BYTES` reject with HTTP 413.
- [evaluate.py:scenario]: `target_scenario` must match a valid `TargetScenario` enum member; invalid values reject with HTTP 422.

## Side Effects
- [evaluate.py:run_voice_pipeline]: Dispatches CPU-bound audio transcription to worker threads and initiates outbound network calls to Groq Cloud.
- [query.py:rag.aquery_corpus]: Queries local vector index in ChromaDB.
- [websocket.py:practice_websocket]: Manages in-memory streaming audio buffers and bi-directional WebSocket message frames.

## Verification
- Run: `pytest`
