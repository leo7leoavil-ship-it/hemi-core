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
import threading
from typing import Any, Optional, Tuple, Union

import numpy as np

from app.core.config import get_settings
from app.core.logging import logger
from app.schemas.response_schemas import ASRMetrics

try:
    import jiwer
except ImportError:
    jiwer = None  # type: ignore


SILENCE_PLACEHOLDER = "[Inaudible / Silence]"
TARGET_SAMPLE_RATE = 16000
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 48000

_NON_WORD_RE = re.compile(r"[^\w\s']", re.UNICODE)
_SPACES_RE = re.compile(r"\s+")


class AudioDurationExceededError(ValueError):
    """Raised when decoded audio duration exceeds MAX_AUDIO_SECONDS."""


class ASRQueueTimeoutError(RuntimeError):
    """Raised when waiting for an available ASR worker slot exceeds ASR_QUEUE_TIMEOUT_SECONDS."""


class ASRCancelledError(RuntimeError):
    """Raised when an in-flight ASR transcription is cooperatively cancelled between segments."""


def pcm16_to_float32(
    pcm_bytes: Union[bytes, bytearray, memoryview], sample_rate: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """Convert raw little-endian 16-bit mono PCM bytes directly to a 16 kHz float32 waveform.

    Eliminates intermediate WAV BytesIO container creation and PyAV re-decoding.
    """
    if not isinstance(sample_rate, int) or not (MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE):
        raise ValueError(
            f"sample_rate must be an integer between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} Hz."
        )
    byte_len = len(pcm_bytes)
    if byte_len == 0:
        raise ValueError("Audio input buffer is empty. Received 0 bytes.")
    if byte_len % 2 != 0:
        raise ValueError("Invalid PCM16 buffer: byte length must be a multiple of 2.")

    pcm_view = np.frombuffer(pcm_bytes, dtype="<i2")
    waveform = pcm_view.astype(np.float32) / 32768.0

    if sample_rate != TARGET_SAMPLE_RATE and waveform.size > 0:
        target_len = max(1, int(round(waveform.size * TARGET_SAMPLE_RATE / float(sample_rate))))
        src_idx = np.linspace(0.0, 1.0, num=waveform.size, endpoint=False)
        dst_idx = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        waveform = np.interp(dst_idx, src_idx, waveform).astype(np.float32)

    return waveform


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
        self.model: Optional[Any] = None
        self._model_lock = threading.Lock()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._semaphore_loop: Optional[asyncio.AbstractEventLoop] = None

        self._initialize_model()
        self._initialized = True

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Return an asyncio.Semaphore bound to the current running event loop."""
        settings = getattr(self, "settings", None) or get_settings()
        max_concurrency = max(
            1, int(getattr(self, "max_concurrency", None) or settings.ASR_MAX_CONCURRENCY)
        )
        loop = asyncio.get_running_loop()
        if (
            getattr(self, "_semaphore", None) is None
            or getattr(self, "_semaphore_loop", None) is not loop
        ):
            self._semaphore = asyncio.Semaphore(max_concurrency)
            self._semaphore_loop = loop
        return self._semaphore  # type: ignore[return-value]

    def _initialize_model(self) -> None:
        """Instantiate WhisperModel with CPU int8 quantization."""
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError:
            WhisperModel = None  # type: ignore

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
        self,
        file_bytes: Union[bytes, np.ndarray],
        expected_transcript: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> ASRMetrics:
        """Decode and transcribe audio bytes or a 16 kHz float32 waveform strictly in memory.

        Args:
            file_bytes: Raw binary audio data (wav, mp3, ogg, webm, m4a) or 1D float32 numpy array.
            expected_transcript: Optional ground-truth text; enables WER/CER scoring.
            cancel_event: Optional threading.Event polled between Whisper segments for cancellation.

        Returns:
            ASRMetrics containing transcript, language, duration, confidence, WPM and
            (optionally) WER/CER.

        Raises:
            AudioDurationExceededError: If audio duration exceeds MAX_AUDIO_SECONDS.
            ASRCancelledError: If cancel_event is set before or during segment iteration.
            ValueError: If audio bytes are empty, exceed MAX_AUDIO_BYTES, or cannot be decoded.
            RuntimeError: If Whisper model is not loaded.
        """
        if file_bytes is None or len(file_bytes) == 0:
            raise ValueError("Audio input buffer is empty. Received 0 bytes.")

        if self.model is None:
            logger.error("ASR Model is not available for transcription.")
            raise RuntimeError(
                "ASR Service model is not initialized or still downloading weights."
            )

        settings = getattr(self, "settings", None) or get_settings()
        max_audio_bytes = int(getattr(self, "max_audio_bytes", None) or settings.MAX_AUDIO_BYTES)
        max_audio_seconds = float(
            getattr(self, "max_audio_seconds", None) or settings.MAX_AUDIO_SECONDS
        )

        if isinstance(file_bytes, np.ndarray):
            waveform = (
                file_bytes.astype(np.float32, copy=False)
                if file_bytes.dtype != np.float32
                else file_bytes
            )
            pcm_duration = float(waveform.size) / float(TARGET_SAMPLE_RATE)
            if pcm_duration > max_audio_seconds:
                raise AudioDurationExceededError(
                    f"Audio duration ({pcm_duration:.2f}s) exceeds maximum allowed "
                    f"duration of {max_audio_seconds:.1f}s."
                )
            audio_source: Any = waveform
            payload_desc = f"{waveform.size} float32 samples ({pcm_duration:.2f}s)"
        else:
            if len(file_bytes) > max_audio_bytes:
                raise ValueError(
                    f"Audio payload size ({len(file_bytes)} bytes) exceeds maximum allowed "
                    f"size of {max_audio_bytes} bytes."
                )
            audio_source = io.BytesIO(file_bytes)
            payload_desc = f"{len(file_bytes)} bytes"

        if cancel_event is not None and cancel_event.is_set():
            raise ASRCancelledError("ASR transcription cancelled before execution started.")

        model_lock = getattr(self, "_model_lock", None)
        try:
            if model_lock is not None:
                model_lock.acquire()
            try:
                if cancel_event is not None and cancel_event.is_set():
                    raise ASRCancelledError("ASR transcription cancelled while waiting for model lock.")

                logger.info(f"Transcribing audio stream ({payload_desc})...")
                segments, info = self.model.transcribe(
                    audio_source,
                    language="en",
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    initial_prompt=getattr(self, "initial_prompt", None) or None,
                )

                total_duration = float(getattr(info, "duration", 0.0) or 0.0)
                if total_duration > max_audio_seconds:
                    raise AudioDurationExceededError(
                        f"Audio duration ({total_duration:.2f}s) exceeds maximum allowed "
                        f"duration of {max_audio_seconds:.1f}s."
                    )

                # Accumulate transcript text and segment statistics
                segment_texts = []
                segment_probs = []
                logprobs = []
                no_speech_probs = []
                speech_seconds = 0.0

                for segment in segments:
                    if cancel_event is not None and cancel_event.is_set():
                        logger.info("Aborting Whisper segment iteration on cancellation signal.")
                        raise ASRCancelledError("ASR transcription cancelled during segment decoding.")

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
            finally:
                if model_lock is not None:
                    model_lock.release()

            full_transcription = " ".join(segment_texts).strip()

            # Calculate confidence score
            if segment_probs:
                confidence = float(sum(segment_probs) / len(segment_probs))
            elif getattr(info, "language_probability", None) is not None:
                confidence = float(info.language_probability)
            else:
                confidence = 0.90

            detected_lang = getattr(info, "language", "en") or "en"
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
        except (AudioDurationExceededError, ASRCancelledError):
            raise
        except Exception as e:
            logger.error(f"Error during in-memory ASR transcription: {e}", exc_info=True)
            raise ValueError(f"Failed to decode or transcribe audio stream: {str(e)}")

    async def atranscribe_audio_stream(
        self,
        file_bytes: Union[bytes, np.ndarray],
        expected_transcript: Optional[str] = None,
    ) -> ASRMetrics:
        """Bounded, timeout-enforced non-blocking wrapper for CPU ASR transcription.

        1. Acquires the per-loop concurrency semaphore within ASR_QUEUE_TIMEOUT_SECONDS
           (raises ASRQueueTimeoutError on queue saturation).
        2. Dispatches CTranslate2 inference to a worker thread with a cooperative
           threading.Event cancellation flag.
        3. Releases the semaphore only when the worker thread finishes, preventing
           concurrent model access even after an async timeout or task cancellation.
        """
        settings = getattr(self, "settings", None) or get_settings()
        queue_timeout = float(
            getattr(self, "queue_timeout_seconds", None) or settings.ASR_QUEUE_TIMEOUT_SECONDS
        )
        asr_timeout = float(
            getattr(self, "asr_timeout_seconds", None) or settings.ASR_TIMEOUT_SECONDS
        )
        semaphore = self._get_semaphore()

        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=queue_timeout)
        except asyncio.TimeoutError as e:
            raise ASRQueueTimeoutError(
                f"ASR engine is busy: queue wait exceeded {queue_timeout:.1f}s."
            ) from e

        cancel_event = threading.Event()
        worker_task = asyncio.create_task(
            asyncio.to_thread(
                self.transcribe_audio_stream,
                file_bytes,
                expected_transcript,
                cancel_event,
            )
        )

        def _on_worker_done(task: "asyncio.Task[ASRMetrics]") -> None:
            semaphore.release()
            if not task.cancelled():
                try:
                    task.exception()
                except Exception:
                    pass

        worker_task.add_done_callback(_on_worker_done)

        try:
            return await asyncio.wait_for(asyncio.shield(worker_task), timeout=asr_timeout)
        except asyncio.TimeoutError:
            cancel_event.set()
            raise
        except asyncio.CancelledError:
            cancel_event.set()
            raise

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
