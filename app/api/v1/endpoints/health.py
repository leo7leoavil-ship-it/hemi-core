"""Health and diagnostic endpoint (spec 4.1).

Reports readiness of ChromaDB, faster-whisper and the Groq configuration.
Protected by the same Bearer / X-API-Key token as the rest of the API.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, status
from app.core.config import get_settings
from app.core.security import verify_api_token
from app.schemas.request_schemas import HealthResponse
from app.services.asr_service import ASRService, get_asr_service
from app.services.llm_service import LLMService, get_llm_service
from app.services.rag_service import RAGService, get_rag_service

router = APIRouter(tags=["Health & Diagnostics"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="System Health & Subsystem Readiness Probe",
    description="Reports in-memory initialization of the ChromaDB, Whisper ASR and Groq LLM subsystems.",
)
async def get_health_status(
    token: str = Depends(verify_api_token),
    rag_service: RAGService = Depends(get_rag_service),
    asr_service: ASRService = Depends(get_asr_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> HealthResponse:
    """Non-blocking health check (no outbound network calls)."""
    settings = get_settings()

    chroma_ready = rag_service.is_ready()
    chroma_detail = "operational" if chroma_ready else "degraded"
    if rag_service.collection is not None:
        try:
            chroma_detail = f"{'operational' if chroma_ready else 'degraded'} ({rag_service.collection.count()} vectors indexed)"
        except Exception:
            chroma_detail = "operational (uncounted)"

    asr_ready = asr_service.is_ready()
    llm_configured = llm_service.is_configured()

    overall_status = "ok" if (chroma_ready and asr_ready and llm_configured) else "degraded"

    return HealthResponse(
        status=overall_status,
        app=settings.PROJECT_NAME,
        version=settings.VERSION,
        chroma_ready=chroma_ready,
        asr_ready=asr_ready,
        llm_configured=llm_configured,
        environment=settings.ENVIRONMENT,
        components={
            "chromadb_vector_store": chroma_detail,
            "whisper_asr_pipeline": (
                f"ready ({settings.ASR_MODEL_NAME}, {settings.ASR_DEVICE}, {settings.ASR_COMPUTE_TYPE})"
                if asr_ready
                else "initializing / not_loaded"
            ),
            "groq_llm_reasoning": (
                f"configured ({settings.LLM_MODEL})" if llm_configured else "missing_api_key"
            ),
        },
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
