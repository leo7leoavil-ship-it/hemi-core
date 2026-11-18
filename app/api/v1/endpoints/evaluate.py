"""Technical English Evaluation Endpoints.

Provides voice (ASR + RAG + LLM) and text (RAG + LLM) evaluation pipelines.
"""

from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from app.core.config import get_settings
from app.core.logging import logger
from app.core.security import verify_api_token
from app.schemas.request_schemas import (
    TargetScenario,
    TextEvaluationRequest,
    resolve_scenario,
)
from app.schemas.response_schemas import EvaluationResponse
from app.services.asr_service import ASRService, get_asr_service
from app.services.evaluation_pipeline import PipelineError, run_text_pipeline, run_voice_pipeline
from app.services.llm_service import LLMService, get_llm_service
from app.services.rag_service import RAGService, get_rag_service

router = APIRouter(prefix="/evaluate", tags=["Technical English Evaluation"])

VALID_AUDIO_EXTENSIONS = (".webm", ".mp3", ".wav", ".ogg", ".m4a")
_SCENARIO_LIST_STR = ", ".join(s.value for s in TargetScenario)


@router.post(
    "/voice",
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Candidate Voice Response (ASR -> RAG -> Groq LLM)",
    description=(
        "Accepts an audio file (multipart/form-data, max 5 MB / 60s), performs memory-resident "
        "faster-whisper transcription, queries the top-3 RAG technical corpus, and evaluates "
        "technical English via Groq Cloud. Supports 9 preset engineering scenarios or "
        "`target_scenario=custom` paired with `custom_scenario` (3-120 printable chars). "
        "If `expected_transcript` is supplied, WER/CER are reported (shadowing / benchmark mode)."
    ),
)
async def evaluate_voice_endpoint(
    audio_file: UploadFile = File(
        ...,
        description="Audio file recording of candidate speech (webm, mp3, wav, ogg, m4a; max 5 MB / 60s)",
    ),
    target_scenario: str = Form(
        default=TargetScenario.SYSTEM_DESIGN.value,
        description=f"Scenario identifier ({_SCENARIO_LIST_STR})",
    ),
    custom_scenario: Optional[str] = Form(
        default=None,
        description=(
            "Custom engineering scenario description (3-120 printable single-line chars). "
            "Required when target_scenario='custom'; must be omitted for preset scenarios."
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
        f"Scenario: '{target_scenario}', Custom: '{custom_scenario}'"
    )

    try:
        scenario, cleaned_custom, _, _ = resolve_scenario(target_scenario, custom_scenario)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
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
            custom_scenario=cleaned_custom,
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
        "from ChromaDB, and returns CEFR rubric scores, terminology detection, and vocabulary "
        "improvements. Supports 9 preset engineering scenarios or `target_scenario=custom` "
        "paired with `custom_scenario` (3-120 printable chars)."
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
        f"Custom: '{request.custom_scenario}', Length: {len(request.candidate_text)} chars"
    )
    try:
        return await run_text_pipeline(
            text=request.candidate_text,
            scenario=request.target_scenario,
            rag=rag,
            llm=llm,
            custom_scenario=request.custom_scenario,
        )
    except PipelineError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
