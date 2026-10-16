"""Technical English Evaluation Endpoints.

Provides voice (ASR + RAG + LLM) and text (RAG + LLM) evaluation pipelines.
"""

from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from app.core.config import get_settings
from app.core.logging import logger
from app.core.security import verify_api_token
from app.schemas.request_schemas import TargetScenario, TextEvaluationRequest
from app.schemas.response_schemas import EvaluationResponse
from app.services.asr_service import ASRService, get_asr_service
from app.services.evaluation_pipeline import PipelineError, run_text_pipeline, run_voice_pipeline
from app.services.llm_service import LLMService, get_llm_service
from app.services.rag_service import RAGService, get_rag_service

router = APIRouter(prefix="/evaluate", tags=["Technical English Evaluation"])

VALID_AUDIO_EXTENSIONS = (".webm", ".mp3", ".wav", ".ogg", ".m4a")


@router.post(
    "/voice",
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Candidate Voice Response (ASR -> RAG -> Groq LLM)",
    description=(
        "Accepts an audio file (multipart/form-data), performs memory-resident faster-whisper "
        "transcription, queries the top-3 RAG technical corpus, and evaluates technical English "
        "via Groq Cloud. If `expected_transcript` is supplied, WER/CER are reported "
        "(shadowing / benchmark mode)."
    ),
)
async def evaluate_voice_endpoint(
    audio_file: UploadFile = File(
        ...,
        description="Audio file recording of candidate speech (webm, mp3, wav, ogg, m4a)",
    ),
    target_scenario: str = Form(
        default=TargetScenario.SYSTEM_DESIGN.value,
        description=(
            "Scenario: system_design_interview, code_review, incident_postmortem, "
            "architecture_presentation"
        ),
    ),
    expected_transcript: Optional[str] = Form(
        default=None,
        description="Optional reference text; enables Word/Character Error Rate computation",
    ),
    token: str = Depends(verify_api_token),
    asr: ASRService = Depends(get_asr_service),
    rag: RAGService = Depends(get_rag_service),
    llm: LLMService = Depends(get_llm_service),
) -> EvaluationResponse:
    """End-to-end voice processing and technical assessment pipeline."""
    settings = get_settings()
    logger.info(
        f"Received voice evaluation request. Filename: '{audio_file.filename}', "
        f"Scenario: '{target_scenario}'"
    )

    try:
        scenario = TargetScenario(target_scenario)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown target_scenario. Valid values: {[s.value for s in TargetScenario]}",
        )

    filename = (audio_file.filename or "").lower()
    if not filename.endswith(VALID_AUDIO_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format. Valid extensions: {list(VALID_AUDIO_EXTENSIONS)}",
        )

    try:
        audio_bytes = await audio_file.read(settings.MAX_AUDIO_BYTES + 1)
    except Exception as e:
        logger.error(f"Failed to read incoming audio stream: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unable to read uploaded audio stream: {str(e)}",
        )

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded audio file is empty (0 bytes received).",
        )
    if len(audio_bytes) > settings.MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio exceeds the maximum allowed size of {settings.MAX_AUDIO_BYTES} bytes.",
        )

    try:
        return await run_voice_pipeline(
            audio_bytes=audio_bytes,
            scenario=scenario,
            asr=asr,
            rag=rag,
            llm=llm,
            expected_transcript=expected_transcript,
        )
    except PipelineError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@router.post(
    "/text",
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Candidate Written or Transcribed Text (RAG -> Groq LLM)",
    description=(
        "Accepts technical text response, retrieves top-3 authoritative engineering snippets "
        "from ChromaDB, and returns CEFR rubric scores, terminology detection, and vocabulary improvements."
    ),
)
async def evaluate_text_endpoint(
    request: TextEvaluationRequest,
    token: str = Depends(verify_api_token),
    rag: RAGService = Depends(get_rag_service),
    llm: LLMService = Depends(get_llm_service),
) -> EvaluationResponse:
    """Text-only technical assessment pipeline."""
    logger.info(
        f"Received text evaluation request. Scenario: '{request.target_scenario.value}', "
        f"Length: {len(request.candidate_text)} chars"
    )
    try:
        return await run_text_pipeline(
            text=request.candidate_text,
            scenario=request.target_scenario,
            rag=rag,
            llm=llm,
        )
    except PipelineError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
