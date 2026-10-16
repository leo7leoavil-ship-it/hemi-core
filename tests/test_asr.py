"""Unit tests for the ASR service (no real Whisper weights are loaded)."""

from types import SimpleNamespace

import pytest

from app.services.asr_service import (
    SILENCE_PLACEHOLDER,
    ASRService,
    compute_error_rates,
    compute_wpm,
    count_words,
    normalize_for_wer,
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
