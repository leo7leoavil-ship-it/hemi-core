# Schemas & Data Contracts

## Overview
Defines Pydantic v2 data models for input request validation, structured evaluation payloads, diagnostic models, and domain enumerations across all engine workflows.

## Interface
- [request_schemas.py:TargetScenario]: Supported evaluation scenario enum (`SYSTEM_DESIGN`, `CODE_REVIEW`, `INCIDENT_POSTMORTEM`, `ARCHITECTURE_PRESENTATION`) with alias normalization.
- [request_schemas.py:SCENARIO_DESCRIPTIONS]: Domain context definitions for technical scenarios.
- [request_schemas.py:TextEvaluationRequest]: Request schema for text assessment (bounded between 2 and 10,000 characters).
- [request_schemas.py:QueryRequest]: Direct semantic search query payload for ChromaDB introspection.
- [request_schemas.py:HealthResponse]: Subsystem diagnostic and readiness report schema.
- [response_schemas.py:FeedbackDetail] / [response_schemas.py:FeedbackItem]: Linguistic feedback schema detailing vocabulary upgrades, grammar corrections, or architectural alignment.
- [response_schemas.py:EvaluationCore]: Quantitative assessment metrics (`overall_score`, `cefr_level`, `lexical_density_ratio`, `grammar_accuracy_score`, `technical_precision_score`, `detected_technical_terms`).
- [response_schemas.py:ASRMetrics]: Speech-to-text telemetry and transcript data (`transcription`, `confidence_score`, `words_per_minute`, `wer_score`, `cer_score`).
- [response_schemas.py:EvaluationResponse]: Comprehensive response payload returned by REST and WebSocket evaluation pipelines.
- [response_schemas.py:RetrievedChunk]: Semantic search hit model containing document chunk, metadata, and distance.
- [response_schemas.py:QueryResponse]: Search result container returned by `/api/v1/query`.

## Invariants
- [request_schemas.py:TextEvaluationRequest.candidate_text]: Character length constrained to `[2, 10000]`.
- [request_schemas.py:QueryRequest.top_k]: Value bounded between 1 and 20.
- [response_schemas.py:EvaluationCore.overall_score]: Bounded between 0.0 and 100.0.
- [response_schemas.py:EvaluationCore.lexical_density_ratio]: Bounded between 0.0 and 1.0.
- [response_schemas.py:ASRMetrics.confidence_score]: Bounded between 0.0 and 1.0.

## Side Effects
- None

## Verification
- Run: `pytest`
