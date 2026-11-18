"""Unit tests for the ASR service (no real Whisper weights are loaded)."""

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.asr_service import (
    SILENCE_PLACEHOLDER,
    ASRCancelledError,
    ASRQueueTimeoutError,
    ASRService,
    AudioDurationExceededError,
    compute_error_rates,
    compute_wpm,
    count_words,
    normalize_for_wer,
    pcm16_to_float32,
)


class FakeWhisper:
    def __init__(self, segments, duration=10.0):
        self.segments = segments
        self.duration = duration
        self.kwargs = None
        self.stream = None

    def transcribe(self, audio, **kwargs):
        self.stream = audio
        self.kwargs = kwargs
        info = SimpleNamespace(language="en", language_probability=0.99, duration=self.duration)
        return iter(self.segments), info


def make_service(model) -> ASRService:
    service = object.__new__(ASRService)
    service.model = model
    service.initial_prompt = "Technical prompt about Kubernetes."
    return service


def seg(text, start, end, avg_logprob=-0.1, no_speech_prob=0.01):
    return SimpleNamespace(
        text=text, start=start, end=end, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob
    )


def test_normalization_and_word_count():
    assert normalize_for_wer("  Hello,   World!! ") == "hello world"
    assert count_words("a b  c") == 3
    assert count_words("   ") == 0


def test_wpm():
    assert compute_wpm(15, 5.4) == 166.7
    assert compute_wpm(10, 0) == 0.0


def test_wer_cer_exact_and_errors():
    assert compute_error_rates("we shard the database", "We shard the database.") == (0.0, 0.0)
    wer, cer = compute_error_rates("we shard the database", "we share the database")
    assert wer == 0.25
    assert 0 < cer < 0.2


def test_wer_without_reference_is_none():
    assert compute_error_rates("", "anything") == (None, None)


def test_transcribe_metrics_and_initial_prompt():
    model = FakeWhisper([seg("We shard the database", 0.0, 2.0), seg("across nodes.", 2.5, 4.0)])
    service = make_service(model)

    metrics = service.transcribe_audio_stream(b"audio-bytes")

    assert metrics.transcription == "We shard the database across nodes."
    assert metrics.word_count == 6
    # 6 words over 3.5 s of detected speech
    assert metrics.speech_duration_seconds == 3.5
    assert metrics.words_per_minute == pytest.approx(102.9, abs=0.1)
    assert 0.9 < metrics.confidence_score <= 1.0
    assert metrics.wer_score is None
    assert model.kwargs["initial_prompt"] == "Technical prompt about Kubernetes."
    assert model.kwargs["language"] == "en"
    assert hasattr(model.stream, "read")  # io.BytesIO: zero disk I/O


def test_transcribe_with_expected_transcript_reports_wer():
    service = make_service(FakeWhisper([seg("we share the database", 0.0, 2.0)]))
    metrics = service.transcribe_audio_stream(b"x", expected_transcript="We shard the database")
    assert metrics.wer_score == 0.25
    assert metrics.cer_score is not None


def test_silence_placeholder():
    service = make_service(FakeWhisper([]))
    assert service.transcribe_audio_stream(b"x").transcription == SILENCE_PLACEHOLDER


def test_empty_buffer_and_missing_model():
    with pytest.raises(ValueError):
        make_service(FakeWhisper([])).transcribe_audio_stream(b"")
    with pytest.raises(RuntimeError):
        make_service(None).transcribe_audio_stream(b"x")


def test_duration_ceiling_short_circuits_before_segment_iteration():
    iterated = False

    def failing_generator():
        nonlocal iterated
        iterated = True
        yield seg("Should never be reached", 0.0, 2.0)

    class LazyWhisper:
        def transcribe(self, audio, **kwargs):
            info = SimpleNamespace(language="en", language_probability=0.99, duration=65.0)
            return failing_generator(), info

    service = make_service(LazyWhisper())
    with pytest.raises(AudioDurationExceededError):
        service.transcribe_audio_stream(b"valid-bytes")
    assert iterated is False


def test_pcm16_to_float32_and_direct_array_transcription():
    samples = np.array([0, 16384, -16384, 32767], dtype="<i2")
    waveform = pcm16_to_float32(samples.tobytes(), sample_rate=16000)
    assert waveform.dtype == np.float32
    assert waveform.shape == (4,)
    assert waveform[1] == pytest.approx(0.5, abs=1e-4)

    # Odd byte length and invalid sample rate are rejected
    with pytest.raises(ValueError):
        pcm16_to_float32(b"\x00\x01\x02", sample_rate=16000)
    with pytest.raises(ValueError):
        pcm16_to_float32(samples.tobytes(), sample_rate=1000)

    # Direct ndarray input skips io.BytesIO
    model = FakeWhisper([seg("Direct PCM array", 0.0, 1.0)], duration=1.0)
    service = make_service(model)
    metrics = service.transcribe_audio_stream(waveform)
    assert metrics.transcription == "Direct PCM array"
    assert isinstance(model.stream, np.ndarray)


def test_cooperative_cancellation_stops_segment_loop():
    cancel_event = threading.Event()
    consumed = 0

    def segment_gen():
        nonlocal consumed
        consumed += 1
        cancel_event.set()
        yield seg("first segment", 0.0, 1.0)
        consumed += 1
        yield seg("second segment", 1.0, 2.0)

    class CancelWhisper:
        def transcribe(self, audio, **kwargs):
            info = SimpleNamespace(language="en", language_probability=0.99, duration=2.0)
            return segment_gen(), info

    service = make_service(CancelWhisper())
    with pytest.raises(ASRCancelledError):
        service.transcribe_audio_stream(b"audio", cancel_event=cancel_event)
    assert consumed == 1


@pytest.mark.asyncio
async def test_async_semaphore_queue_timeout():
    gate = threading.Event()
    started = threading.Event()

    class SlowWhisper:
        def transcribe(self, audio, **kwargs):
            started.set()
            gate.wait(timeout=2.0)
            info = SimpleNamespace(language="en", language_probability=0.99, duration=1.0)
            return iter([seg("done", 0.0, 1.0)]), info

    service = make_service(SlowWhisper())
    service.max_concurrency = 1
    service.queue_timeout_seconds = 0.05
    service.asr_timeout_seconds = 5.0

    task1 = asyncio.create_task(service.atranscribe_audio_stream(b"first"))
    await asyncio.to_thread(started.wait, 1.0)

    with pytest.raises(ASRQueueTimeoutError):
        await service.atranscribe_audio_stream(b"second")

    gate.set()
    result1 = await task1
    assert result1.transcription == "done"


def test_pyav_19_metadata_errors_compat_patch():
    import io
    import wave
    import faster_whisper.audio as fwa
    from app.services.asr_service import _patch_pyav_metadata_errors

    _patch_pyav_metadata_errors()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    buf.seek(0)
    arr = fwa.decode_audio(buf)
    assert arr.shape == (1600,)
    assert arr.dtype == np.float32
