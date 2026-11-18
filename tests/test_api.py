"""Integration tests for the FastAPI MaaS endpoints (REST + WebSocket)."""

import pytest
from tests.conftest import AUTH_HEADERS

VOICE_URL = "/api/v1/evaluate/voice"
TEXT_URL = "/api/v1/evaluate/text"
WS_URL = "/api/v1/ws/practice"


def _voice(client, filename="sample.wav", content=b"RIFFfakeaudio", **form):
    data = {"target_scenario": "system_design_interview", **form}
    return client.post(
        VOICE_URL,
        headers=AUTH_HEADERS,
        files={"audio_file": (filename, content, "audio/wav")},
        data=data,
    )


# ---------------------------------------------------------------- root & health
def test_root_index(client):
    data = client.get("/").json()
    assert data["service"] == "Tech Tutor Engine"
    assert data["status"] == "online"
    assert data["websocket"] == "/api/v1/ws/practice"


@pytest.mark.parametrize("path", ["/health", "/api/v1/health"])
def test_health_contract(client, path):
    response = client.get(path, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "Tech Tutor Engine"
    assert data["chroma_ready"] is True
    assert data["asr_ready"] is True
    assert data["llm_configured"] is True
    assert "213439 vectors" in data["components"]["chromadb_vector_store"]


def test_health_requires_token(client):
    assert client.get("/health").status_code == 401


# ------------------------------------------------------------------------ auth
def test_wrong_token_rejected(client):
    response = client.get("/health", headers={"Authorization": "Bearer nope"})
    assert response.status_code == 401


def test_x_api_key_accepted(client):
    assert client.get("/health", headers={"X-API-Key": "test-token"}).status_code == 200


# ----------------------------------------------------------------------- query
def test_query_contract(client, fakes):
    response = client.post(
        "/api/v1/query",
        headers=AUTH_HEADERS,
        json={"query_text": "How does horizontal database sharding mitigate write bottleneck?", "top_k": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["top_k"] == 3
    chunk = data["chunks"][0]
    assert set(chunk) == {"document", "metadata", "distance", "source"}
    assert chunk["distance"] == 0.31


def test_query_validation(client):
    response = client.post(
        "/api/v1/query", headers=AUTH_HEADERS, json={"query_text": "valid text", "top_k": 0}
    )
    assert response.status_code == 422


def test_query_unauthorized(client):
    assert client.post("/api/v1/query", json={"query_text": "sharding"}).status_code == 401


# ------------------------------------------------------------------- text eval
def test_evaluate_text_unauthorized(client):
    response = client.post(TEXT_URL, json={"candidate_text": "We use a cache.", "target_scenario": "code_review"})
    assert response.status_code == 401


def test_evaluate_text_ok_and_scenario_fusion(client, fakes):
    response = client.post(
        TEXT_URL,
        headers=AUTH_HEADERS,
        json={"candidate_text": "We implemented a Redis cache to mitigate latency.", "target_scenario": "code_review"},
    )
    assert response.status_code == 200
    core = response.json()["evaluation_core"]
    assert core["cefr_level"] == "C1"
    assert core["technical_precision_score"] == 89.0
    assert fakes["rag"].calls[-1]["scenario"] == "code_review"


def test_evaluate_text_accepts_legacy_scenario_label(client, fakes):
    response = client.post(
        TEXT_URL,
        headers=AUTH_HEADERS,
        json={"candidate_text": "We introduced a message broker.", "target_scenario": "System Design"},
    )
    assert response.status_code == 200
    assert fakes["rag"].calls[-1]["scenario"] == "system_design_interview"


def test_evaluate_text_llm_failure_is_502(client, fakes):
    fakes["llm"].fail = True
    response = client.post(
        TEXT_URL,
        headers=AUTH_HEADERS,
        json={"candidate_text": "We introduced a message broker.", "target_scenario": "code_review"},
    )
    assert response.status_code == 502


# ------------------------------------------------------------------ voice eval
def test_voice_happy_path(client, fakes):
    response = _voice(client)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["asr_metrics"]["word_count"] == 11
    assert data["asr_metrics"]["wer_score"] is None
    assert fakes["rag"].calls[-1]["scenario"] == "system_design_interview"


def test_voice_expected_transcript_enables_wer(client, fakes):
    response = _voice(client, expected_transcript="We shard the database.")
    assert response.status_code == 200
    assert response.json()["asr_metrics"]["wer_score"] == 0.05
    assert fakes["asr"].received_expected == "We shard the database."


def test_voice_rejects_unsupported_extension(client):
    assert _voice(client, filename="notes.txt").status_code == 400


def test_voice_rejects_empty_file(client):
    assert _voice(client, content=b"").status_code == 400


def test_voice_rejects_unknown_scenario(client):
    assert _voice(client, target_scenario="cooking_class").status_code == 422


def test_voice_silence_is_422(client, fakes):
    fakes["asr"].transcription = "[Inaudible / Silence]"
    assert _voice(client).status_code == 422


def test_voice_requires_token(client):
    response = client.post(
        VOICE_URL,
        files={"audio_file": ("a.wav", b"abc", "audio/wav")},
        data={"target_scenario": "code_review"},
    )
    assert response.status_code == 401


# ------------------------------------------------------------------- websocket
def test_websocket_rejects_missing_token(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(WS_URL):
            pass


def test_websocket_full_session(client):
    with client.websocket_connect(f"{WS_URL}?token=test-token") as ws:
        assert ws.receive_json() == {"event": "ready"}
        ws.send_text('{"action": "config", "target_scenario": "incident_postmortem"}')
        ws.send_bytes(b"chunk-1")
        ws.send_bytes(b"chunk-2")
        ws.send_text('{"action": "end"}')

        events = [ws.receive_json() for _ in range(4)]
        assert [e["event"] for e in events] == ["transcribing", "retrieving", "evaluating", "completed"]
        assert events[1]["partial_text"].startswith("We shard the database")
        assert events[3]["payload"]["evaluation_core"]["cefr_level"] == "C1"


def test_websocket_empty_utterance_reports_error_and_stays_open(client):
    with client.websocket_connect(f"{WS_URL}?token=test-token") as ws:
        ws.receive_json()
        ws.send_text('{"action": "end"}')
        error = ws.receive_json()
        assert error["event"] == "error" and error["status_code"] == 400

        ws.send_text('{"action": "nonsense"}')
        assert ws.receive_json()["status_code"] == 400


def test_voice_queue_saturation_returns_429(client, fakes, monkeypatch):
    from app.services.asr_service import ASRQueueTimeoutError

    async def busy_asr(file_bytes, expected_transcript=None):
        raise ASRQueueTimeoutError("ASR engine is busy: queue wait exceeded 10.0s.")

    monkeypatch.setattr(fakes["asr"], "atranscribe_audio_stream", busy_asr)
    response = _voice(client)
    assert response.status_code == 429


def test_voice_stage_timeout_returns_504(client, fakes, monkeypatch):
    import asyncio

    async def timeout_asr(file_bytes, expected_transcript=None):
        raise asyncio.TimeoutError("stage timed out")

    monkeypatch.setattr(fakes["asr"], "atranscribe_audio_stream", timeout_asr)
    response = _voice(client)
    assert response.status_code == 504


def test_websocket_pcm16_direct_array_and_duration_limit(client):
    with client.websocket_connect(f"{WS_URL}?token=test-token") as ws:
        assert ws.receive_json() == {"event": "ready"}
        ws.send_text('{"action": "config", "format": "pcm16", "sample_rate": 16000}')
        # Odd byte length PCM16 should return 422
        ws.send_bytes(b"\x00\x01\x02")
        ws.send_text('{"action": "end"}')
        err = ws.receive_json()
        assert err["event"] == "error" and err["status_code"] == 422

        # Valid 16-bit PCM samples (even byte count)
        ws.send_bytes(b"\x00\x00\x10\x00" * 100)
        ws.send_text('{"action": "end"}')
        events = [ws.receive_json() for _ in range(4)]
        assert [e["event"] for e in events] == ["transcribing", "retrieving", "evaluating", "completed"]


def test_custom_scenario_across_text_voice_and_websocket(client, fakes):
    # 1. Text endpoint with custom scenario forwards resolved topic to RAG
    res_text = client.post(
        TEXT_URL,
        headers=AUTH_HEADERS,
        json={
            "candidate_text": "We inspected the PodDisruptionBudget and readiness probe.",
            "target_scenario": "custom",
            "custom_scenario": "Kubernetes zero-downtime deployment review",
        },
    )
    assert res_text.status_code == 200
    assert fakes["rag"].calls[-1]["scenario"] == "Kubernetes zero-downtime deployment review"

    # Missing custom_scenario when target_scenario="custom" -> 422
    res_missing = client.post(
        TEXT_URL,
        headers=AUTH_HEADERS,
        json={
            "candidate_text": "We inspected the PodDisruptionBudget.",
            "target_scenario": "custom",
        },
    )
    assert res_missing.status_code == 422

    # 2. Voice endpoint with custom scenario and expanded preset (devops_ci_cd)
    res_voice_preset = _voice(client, target_scenario="devops_ci_cd")
    assert res_voice_preset.status_code == 200
    assert fakes["rag"].calls[-1]["scenario"] == "devops_ci_cd"

    res_voice_custom = _voice(
        client,
        target_scenario="custom",
        custom_scenario="GraphQL federation gateway caching",
    )
    assert res_voice_custom.status_code == 200
    assert fakes["rag"].calls[-1]["scenario"] == "GraphQL federation gateway caching"

    # Providing custom_scenario with a preset is rejected with 422
    res_invalid_pair = _voice(
        client,
        target_scenario="code_review",
        custom_scenario="Should not be allowed with preset",
    )
    assert res_invalid_pair.status_code == 422

    # 3. WebSocket custom scenario and preset reset
    with client.websocket_connect(f"{WS_URL}?token=test-token") as ws:
        assert ws.receive_json() == {"event": "ready"}
        ws.send_text(
            '{"action": "config", "target_scenario": "custom", '
            '"custom_scenario": "PostgreSQL vacuum and MVCC tuning"}'
        )
        ws.send_bytes(b"audio-chunk")
        ws.send_text('{"action": "end"}')
        events = [ws.receive_json() for _ in range(4)]
        assert events[-1]["event"] == "completed"
        assert fakes["rag"].calls[-1]["scenario"] == "PostgreSQL vacuum and MVCC tuning"

        # Switching back to a preset clears custom_scenario automatically
        ws.send_text('{"action": "config", "target_scenario": "daily_standup"}')
        ws.send_bytes(b"audio-chunk-2")
        ws.send_text('{"action": "end"}')
        events2 = [ws.receive_json() for _ in range(4)]
        assert events2[-1]["event"] == "completed"
        assert fakes["rag"].calls[-1]["scenario"] == "daily_standup"


def test_interactive_ui_console_and_openapi_guide(client):
    ui_res = client.get("/ui")
    assert ui_res.status_code == 200
    assert "text/html" in ui_res.headers["content-type"]
    html = ui_res.text
    assert "API_AUTH_TOKEN" in html
    assert "system_design_interview" in html
    assert "devops_ci_cd" in html
    assert "custom_scenario" in html

    openapi_res = client.get("/openapi.json")
    assert openapi_res.status_code == 200
    info_desc = openapi_res.json()["info"]["description"]
    assert "API_AUTH_TOKEN" in info_desc
    assert "Postman" in info_desc
    assert "/api/v1/ws/practice" in info_desc
    assert "devops_ci_cd" in info_desc
