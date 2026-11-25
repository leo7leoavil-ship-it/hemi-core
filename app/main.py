"""hemi-core - Main FastAPI Application Entry Point.

Clean Architecture implementation of the Model-as-a-Service (MaaS) evaluation engine:
Lifespan context management (corpus provisioning + AI model preloading), CORS middleware
and the v1 API (REST + WebSocket). An optional Gradio sandbox can be mounted at /ui by
setting ENABLE_UI=true (disabled by default; it is outside the MaaS scope).
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.api.v1.endpoints.health import get_health_status
from app.api.v1.router import api_v1_router
from app.core.config import get_settings
from app.core.logging import logger
from app.core.openapi_docs import (
    OPENAPI_TAGS,
    build_openapi_description,
    build_oracle_docs_html,
    build_redoc_html,
)
from app.services.asr_service import get_asr_service
from app.services.corpus_sync import ensure_corpus_available
from app.services.llm_service import get_llm_service
from app.services.rag_service import get_rag_service
from app.ui.gradio_app import ui_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI Lifespan Context Manager.

    1. Provisions the persisted vector DB (HF Hub dataset) if it is missing locally.
    2. Preloads heavy AI models (ChromaDB PersistentClient, SentenceTransformer,
       faster-whisper int8, Groq Async Client) to minimize first-request latency.

    Blocking work runs in worker threads so the event loop stays responsive.
    """
    settings = get_settings()
    logger.info(f"=== Starting {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENVIRONMENT}] ===")

    if not settings.API_AUTH_TOKEN:
        logger.warning(
            "API_AUTH_TOKEN is not set: every authenticated request will be rejected (HTTP 401)."
        )

    try:
        # 1. Vector DB must exist BEFORE the RAG service opens the PersistentClient.
        await asyncio.to_thread(ensure_corpus_available, settings)

        logger.info("Initializing & caching RAG vector store singleton...")
        rag_service = await asyncio.to_thread(get_rag_service)
        logger.info(f"RAG Service initialized: ready={rag_service.is_ready()}")

        logger.info("Initializing & caching Whisper ASR singleton in memory...")
        asr_service = await asyncio.to_thread(get_asr_service)
        logger.info(f"ASR Service initialized: ready={asr_service.is_ready()}")

        logger.info("Initializing Groq Cloud LLM reasoning client...")
        llm_service = get_llm_service()
        logger.info(f"LLM Service initialized: configured={llm_service.is_configured()}")

        logger.info("All AI pipelines preloaded in memory. System ready for inference.")
    except Exception as e:
        logger.error(f"Error during lifespan startup preloading: {e}", exc_info=True)

    yield

    logger.info("=== Gracefully shutting down hemi-core ===")


def create_application() -> FastAPI:
    """Application factory for hemi-core."""
    settings = get_settings()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=build_openapi_description(settings.DESCRIPTION),
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    @app.get("/docs", include_in_schema=False, response_class=HTMLResponse)
    async def custom_api_docs_html() -> HTMLResponse:
        return HTMLResponse(
            content=build_oracle_docs_html(
                project_name=settings.PROJECT_NAME,
                version=settings.VERSION,
                description=settings.DESCRIPTION,
            )
        )

    @app.get("/redoc", include_in_schema=False, response_class=HTMLResponse)
    async def custom_redoc_html() -> HTMLResponse:
        return HTMLResponse(
            content=build_redoc_html(
                project_name=settings.PROJECT_NAME,
                openapi_url="/openapi.json",
            )
        )

    # 1. CORS Middleware (credentials are only valid with explicit origins, not "*")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials="*" not in settings.ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Root health check endpoint directly at /health (spec 4.1; token protected)
    app.add_api_route(
        "/health",
        get_health_status,
        methods=["GET"],
        tags=["Health & Diagnostics"],
        summary="Root Health Check",
    )

    # 3. Root informational endpoint
    @app.get(
        "/",
        tags=["System Information"],
        summary="Service Root Index",
        response_class=JSONResponse,
    )
    async def root_index() -> Dict[str, str]:
        return {
            "service": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "status": "online",
            "documentation": "/docs",
            "interactive_console": "/ui",
            "health_check": "/health",
            "api_v1_base": "/api/v1",
            "websocket": "/api/v1/ws/practice",
        }

    # 4. API v1 routers (/query, /evaluate/voice, /evaluate/text, /ws/practice, /health)
    app.include_router(api_v1_router, prefix="/api/v1")

    # 5. Zero-dependency Interactive Testing Console at /ui
    app.include_router(ui_router)

    return app


app = create_application()


if __name__ == "__main__":
    import uvicorn
    uvicorn_settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=uvicorn_settings.HOST,
        port=uvicorn_settings.PORT,
        reload=uvicorn_settings.DEBUG,
    )
