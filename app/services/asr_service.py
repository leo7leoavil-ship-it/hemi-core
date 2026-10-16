"""ASR (Automatic Speech Recognition) Pipeline Service.

Executes memory-resident audio transcription using faster-whisper (large-v3-turbo, int8)
directly on io.BytesIO streams without temporary disk I/O.

Metrics produced per request:
* ``confidence_score``  - mean exp(avg_logprob) over segments, in [0, 1].
* ``words_per_minute``  - speech rate computed over the detected speech time.
* ``wer_score`` / ``cer_score`` - only when ``expected_transcript`` is supplied
  (shadowing / benchmark mode), computed with ``jiwer``.
"""

import asyncio
import io
import math
import re
from typing import Optional, Tuple
from app.core.config import get_settings
from app.core.logging import logger
from app.schemas.response_schemas import ASRMetrics

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore

try:
    import jiwer
except ImportError:
    jiwer = None  # type: ignore


SILENCE_PLACEHOLDER = "[Inaudible / Silence]"

_NON_WORD_RE = re.compile(r"[^\w\s']", re.UNICODE)
_SPACES_RE = re.compile(r"\s+")


def normalize_for_wer(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace before scoring."""
    text = _NON_WORD_RE.sub(" ", text.lower())
    return _SPACES_RE.sub(" ", text).strip()


def compute_error_rates(
    reference: str, hypothesis: str
) -> Tuple[Optional[float], Optional[float]]:
    """Return (WER, CER) of ``hypothesis`` against ``reference`` using jiwer.

    Returns ``(None, None)`` if the reference is empty or jiwer is unavailable.
    """
    ref = normalize_for_wer(reference or "")
    hyp = normalize_for_wer(hypothesis or "")
    if not ref:
        return None, None
    if jiwer is None:
        logger.warning("jiwer is not installed; WER/CER cannot be computed.")
        return None, None
    return round(float(jiwer.wer(ref, hyp)), 4), round(float(jiwer.cer(ref, hyp)), 4)


def count_words(text: str) -> int:
    """Count whitespace-separated word tokens."""
    return len(text.split()) if text and text.strip() else 0


def compute_wpm(word_count: int, speech_seconds: float) -> float:
    """Words per minute over ``speech_seconds`` (0.0 if there is no speech time)."""
    if speech_seconds <= 0 or word_count <= 0:
        return 0.0
    return round(word_count / speech_seconds * 60.0, 1)


class ASRService:
    """Singleton service for in-memory speech-to-text inference with faster-whisper."""

    _instance: Optional["ASRService"] = None

    def __new__(cls) -> "ASRService":
        if cls._instance is None:
            cls._instance = super(ASRService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.settings = get_settings()
        self.model_name = self.settings.ASR_MODEL_NAME
        self.device = self.settings.ASR_DEVICE
        self.compute_type = self.settings.ASR_COMPUTE_TYPE
        self.initial_prompt = self.settings.ASR_INITIAL_PROMPT
        self.model: Optional[WhisperModel] = None

        self._initialize_model()
        self._initialized = True

    def _initialize_model(self) -> None:
        """Instantiate WhisperModel with CPU int8 quantization."""
        if WhisperModel is None:
            logger.warning("faster-whisper package is not installed yet.")
            return

        try:
            logger.info(
                f"Loading faster-whisper model '{self.model_name}' on device '{self.device}' "
                f"with compute_type '{self.compute_type}'..."
            )
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.settings.ASR_CPU_THREADS,
                num_workers=1,
            )
            logger.info(f"faster-whisper '{self.model_name}' model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize WhisperModel '{self.model_name}': {e}", exc_info=True)
            self.model = None

    def transcribe_audio_stream(
        self, file_bytes: bytes, expected_transcript: Optional[str] = None
    ) -> ASRMetrics:
        """Decode and transcribe audio bytes strictly in memory using io.BytesIO.
        
        Args:
            file_bytes: Raw binary audio data (wav, mp3, ogg, webm, etc.).
            expected_transcript: Optional ground-truth text; enables WER/CER scoring.

        Returns:
            ASRMetrics containing transcript, language, duration, confidence, WPM and
            (optionally) WER/CER.

        Raises:
            ValueError: If audio bytes are empty or unprocessable.
            RuntimeError: If Whisper model is not loaded.
        """
        if not file_bytes or len(file_bytes) == 0:
            raise ValueError("Audio input buffer is empty. Received 0 bytes.")

        if self.model is None:
            # If model is not loaded yet (e.g. during offline test), provide descriptive error
            logger.error("ASR Model is not available for transcription.")
            raise RuntimeError(
                "ASR Service model is not initialized or still downloading weights."
            )

        try:
            # Memory stream decoding - strictly 0 disk I/O
            audio_stream = io.BytesIO(file_bytes)

            logger.info(f"Transcribing audio stream of size {len(file_bytes)} bytes...")
            segments, info = self.model.transcribe(
                audio_stream,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                initial_prompt=self.initial_prompt or None,
            )

            # Accumulate transcript text and segment statistics
            segment_texts = []
            segment_probs = []
            logprobs = []
            no_speech_probs = []
            speech_seconds = 0.0

            for segment in segments:
                text = segment.text.strip()
                if text:
                    segment_texts.append(text)
                start = getattr(segment, "start", None)
                end = getattr(segment, "end", None)
                if start is not None and end is not None and end > start:
                    speech_seconds += float(end) - float(start)
                avg_logprob = getattr(segment, "avg_logprob", None)
                if avg_logprob is not None:
                    logprobs.append(float(avg_logprob))
                    try:
                        prob = math.exp(avg_logprob)
                        segment_probs.append(min(1.0, max(0.0, prob)))
                    except OverflowError:
                        segment_probs.append(0.5)
                nsp = getattr(segment, "no_speech_prob", None)
                if nsp is not None:
                    no_speech_probs.append(float(nsp))

            full_transcription = " ".join(segment_texts).strip()

            # Calculate confidence score
            if segment_probs:
                confidence = float(sum(segment_probs) / len(segment_probs))
            elif getattr(info, "language_probability", None) is not None:
                confidence = float(info.language_probability)
            else:
                confidence = 0.90

            detected_lang = getattr(info, "language", "en") or "en"
            total_duration = float(getattr(info, "duration", 0.0) or 0.0)
            # Prefer actual speech time (VAD-filtered) for the speech rate.
            duration = speech_seconds if speech_seconds > 0 else total_duration

            word_count = count_words(full_transcription)
            wer_score: Optional[float] = None
            cer_score: Optional[float] = None
            if expected_transcript and expected_transcript.strip():
                wer_score, cer_score = compute_error_rates(
                    expected_transcript, full_transcription
                )

            logger.info(
                f"ASR complete. Language: {detected_lang}, Speech: {duration:.2f}s, "
                f"Confidence: {confidence:.2f}, Words: {word_count}, WER: {wer_score}"
            )

            return ASRMetrics(
                transcription=full_transcription if full_transcription else SILENCE_PLACEHOLDER,
                detected_language=detected_lang,
                speech_duration_seconds=round(duration, 2),
                confidence_score=round(confidence, 4),
                word_count=word_count,
                words_per_minute=compute_wpm(word_count, duration),
                wer_score=wer_score,
                cer_score=cer_score,
                avg_logprob=round(sum(logprobs) / len(logprobs), 4) if logprobs else None,
                no_speech_prob=(
                    round(min(1.0, sum(no_speech_probs) / len(no_speech_probs)), 4)
                    if no_speech_probs
                    else None
                ),
            )
        except Exception as e:
            logger.error(f"Error during in-memory ASR transcription: {e}", exc_info=True)
            raise ValueError(f"Failed to decode or transcribe audio stream: {str(e)}")

    async def atranscribe_audio_stream(
        self, file_bytes: bytes, expected_transcript: Optional[str] = None
    ) -> ASRMetrics:
        """Non-blocking wrapper: runs the CPU-bound CTranslate2 inference in a worker thread."""
        return await asyncio.to_thread(
            self.transcribe_audio_stream, file_bytes, expected_transcript
        )

    def is_ready(self) -> bool:
        """Check if ASR model is loaded and ready for inference."""
        return self.model is not None


_asr_singleton: Optional[ASRService] = None


def get_asr_service() -> ASRService:
    """Retrieve singleton instance of ASRService."""
    global _asr_singleton
    if _asr_singleton is None:
        _asr_singleton = ASRService()
    return _asr_singleton
