# API v1 Router

## Overview
Aggregates version 1 API endpoints into a unified FastAPI router, organizing health probes, semantic queries, multimodal evaluations, and real-time WebSocket sessions under the `/api/v1` prefix.

## Interface
- [router.py:api_v1_router]: FastAPI `APIRouter` aggregating health, query, evaluate, and websocket endpoint routers.
- [Endpoints](./endpoints/README.md): Subdirectory contract for individual REST and WebSocket route handlers.

## Invariants
- [router.py:api_v1_router]: Preserves route prefixing and tags defined across child routers.

## Side Effects
- None

## Verification
- Run: `pytest`
