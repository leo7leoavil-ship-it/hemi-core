# Application Core

## Overview
Central application package hosting the FastAPI application factory, lifespan context management, CORS middleware configuration, and modular routing across core, API, schemas, services, and optional UI.

## Interface
- [main.py:create_application]: Application factory instantiating and configuring FastAPI, middleware, routes, and error handling.
- [main.py:lifespan]: Async lifespan manager orchestrating startup preloading (corpus download, ChromaDB client, SentenceTransformer, faster-whisper, Groq) and graceful shutdown.
- [main.py:app]: Exported ASGI application object.
- [Core](./core/README.md): Subdirectory contract for configuration, security, and logging.
- [API](./api/README.md): Subdirectory contract for versioned HTTP REST and WebSocket routes.
- [Schemas](./schemas/README.md): Subdirectory contract for Pydantic request/response models.
- [Services](./services/README.md): Subdirectory contract for ASR, RAG, LLM, and pipeline orchestration.
- [UI](./ui/README.md): Subdirectory contract for the optional Gradio web sandbox.

## Invariants
- [main.py:lifespan]: Corpus provisioning via `ensure_corpus_available` must run before opening the ChromaDB `PersistentClient`.
- [main.py:create_application]: Emits a warning log at startup if `API_AUTH_TOKEN` is unset.

## Side Effects
- Binds ASGI application on configured host and port via Uvicorn.
- Initializes heavy neural models in memory.

## Verification
- Run: `pytest`
