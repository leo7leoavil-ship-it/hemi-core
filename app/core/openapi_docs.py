"""API Documentation Portal (/docs), ReDoc Theme (/redoc), and OpenAPI Description Builder.

Provides:
1. A clean, Oracle-inspired neutral corporate documentation portal at `/docs`
   focused strictly on consuming the exposed REST and WebSocket services (no
   Swagger UI clutter, no purple accents, copyable cURL/Postman recipes).
2. A clean Markdown OpenAPI description for `/openapi.json` and `/redoc` without
   horizontal rules or backticks in H2 headers so ReDoc renders every section
   completely with zero empty panels.
3. Shared corporate SVG favicon (`FAVICON_DATA_URI`).
"""

from html import escape
from typing import Any, Dict, List

from app.schemas.request_schemas import (
    CUSTOM_SCENARIO_MAX_LEN,
    CUSTOM_SCENARIO_MIN_LEN,
    SCENARIO_DESCRIPTIONS,
    SCENARIO_TITLES,
    TargetScenario,
)

FAVICON_DATA_URI = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E"
    "%3Crect width='64' height='64' rx='14' fill='%231b365d'/%3E"
    "%3Cpath d='M18 36v-8m7 14V22m7 24V18m7 24V22m7 14v-8' stroke='%23e2e8f0' "
    "stroke-width='4' stroke-linecap='round'/%3E%3C/svg%3E"
)

OPENAPI_TAGS: List[Dict[str, Any]] = [
    {
        "name": "System Information",
        "description": "Service discovery endpoint returning active REST, WebSocket, documentation, and testing console paths.",
    },
    {
        "name": "Health & Diagnostics",
        "description": "Operational readiness probe for ChromaDB vector store, faster-whisper ASR, and Groq Cloud LLM.",
    },
    {
        "name": "Technical English Evaluation",
        "description": "Voice (`POST /api/v1/evaluate/voice`) and text (`POST /api/v1/evaluate/text`) CEFR evaluation endpoints.",
    },
    {
        "name": "Semantic Retrieval",
        "description": "Direct cosine-similarity vector search (`POST /api/v1/query`) over the technical engineering corpus.",
    },
    {
        "name": "Interactive Testing Console",
        "description": "Browser-based testing interface (`/ui`) for live microphone recording, WebSocket streaming, and REST verification.",
    },
]


def build_openapi_description(base_summary: str) -> str:
    """Build a ReDoc-safe Markdown OpenAPI reference (no `---` or backticks in H2 headings)."""
    scenario_rows = []
    for member in TargetScenario:
        title = SCENARIO_TITLES.get(member, member.value)
        desc = SCENARIO_DESCRIPTIONS.get(member, "")
        scenario_rows.append(f"| `{member.value}` | **{title}** | {desc} |")
    scenario_table = "\n".join(scenario_rows)

    return f"""{base_summary}

Use this specification to integrate external clients (Postman, cURL, backend services, or web frontends) with the hemi-core REST and WebSocket APIs. For live browser testing with microphone capture, open the [Interactive Testing Console](/ui).

## 1. Service Overview and Base Endpoints

hemi-core exposes synchronous HTTP REST endpoints under `/api/v1` and a full-duplex WebSocket channel for real-time voice practice.

| Resource | Path | Purpose |
| :--- | :--- | :--- |
| **REST API Base** | `/api/v1` | JSON and `multipart/form-data` evaluation and retrieval endpoints |
| **WebSocket Stream** | `/api/v1/ws/practice` | Full-duplex audio streaming with real-time stage events |
| **Health Probe** | `/health` and `/api/v1/health` | Subsystem readiness (`chroma_ready`, `asr_ready`, `llm_configured`) |
| **API Guide Portal** | `/docs` | Structured Oracle-style consumption guide |
| **Interactive Console** | `/ui` | Browser testing console with microphone capture and Light/Dark themes |

## 2. Authentication and Authorization

All service endpoints (`/health`, `/api/v1/*`, and `WS /api/v1/ws/practice`) enforce fail-closed token validation against the server's `API_AUTH_TOKEN` (alias `API_TOKEN` in `.env`). Requests without a valid token receive `HTTP 401 Unauthorized` (or WebSocket close code `1008`).

| Client Type | Header or Parameter | Configuration |
| :--- | :--- | :--- |
| **Postman (Bearer)** | Authorization → Bearer Token | Token value: `<API_AUTH_TOKEN>` |
| **Postman / HTTP Header** | `X-API-Key` | `X-API-Key: <API_AUTH_TOKEN>` |
| **cURL** | `-H "Authorization: Bearer ..."` | `-H "Authorization: Bearer <API_AUTH_TOKEN>"` |
| **WebSocket Client** | Query Parameter `?token=` | `wss://<host>/api/v1/ws/practice?token=<API_AUTH_TOKEN>` |

## 3. Scenario Reference

Evaluation endpoints accept 9 preset software engineering scenarios plus an explicit `custom` mode paired with `custom_scenario`.

| Slug (`target_scenario`) | Display Title | Domain Rubric Context |
| :--- | :--- | :--- |
{scenario_table}

**Custom Scenario Rules:**
1. Set `target_scenario` to `"custom"`.
2. Provide `custom_scenario` with **{CUSTOM_SCENARIO_MIN_LEN} to {CUSTOM_SCENARIO_MAX_LEN} printable single-line characters** (for example: `"Kubernetes cluster DNS and CNI troubleshooting"`).
3. Omit `custom_scenario` (or set it to `null`) when using any of the 9 preset scenarios; sending both returns `HTTP 422`.

## 4. REST Consumption Examples

The following examples show how to invoke each endpoint using cURL or Postman.

### Health Check (`GET /health`)
```bash
curl -X GET "https://<host>/health" \\
  -H "Authorization: Bearer <API_AUTH_TOKEN>"
```

### Text Evaluation (`POST /api/v1/evaluate/text`)
In Postman, set method to `POST`, select **Body → raw → JSON**, and add header `Content-Type: application/json`.
```bash
curl -X POST "https://<host>/api/v1/evaluate/text" \\
  -H "Authorization: Bearer <API_AUTH_TOKEN>" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "candidate_text": "We implemented horizontal database sharding and a Redis cache layer to mitigate write latency.",
    "target_scenario": "system_design_interview"
  }}'
```

### Voice Evaluation (`POST /api/v1/evaluate/voice`)
In Postman, set method to `POST` and select **Body → form-data** with `audio_file` (File, max 5 MB / 60 s), `target_scenario` (Text), optional `custom_scenario` (Text, required only when `target_scenario=custom`), and optional `expected_transcript` (Text, enables WER/CER).
```bash
curl -X POST "https://<host>/api/v1/evaluate/voice" \\
  -H "Authorization: Bearer <API_AUTH_TOKEN>" \\
  -F "audio_file=@sample.wav" \\
  -F "target_scenario=code_review"
```

### Semantic Corpus Query (`POST /api/v1/query`)
```bash
curl -X POST "https://<host>/api/v1/query" \\
  -H "Authorization: Bearer <API_AUTH_TOKEN>" \\
  -H "Content-Type: application/json" \\
  -d '{{"query_text": "How does horizontal database sharding mitigate write bottlenecks?", "top_k": 3}}'
```

## 5. Real-Time WebSocket Protocol

Connect to `wss://<host>/api/v1/ws/practice?token=<API_AUTH_TOKEN>`. Upon handshake acceptance, the server emits `{{"event": "ready"}}`.

1. **Configure session (optional JSON text frame before each utterance):**
```json
{{
  "action": "config",
  "target_scenario": "custom",
  "custom_scenario": "Distributed tracing and latency debugging",
  "format": "webm",
  "sample_rate": 16000,
  "expected_transcript": ""
}}
```
2. **Stream audio:** Send one or more binary WebSocket frames (max 5 MB / 60 s cumulative). `format` accepts `"webm"` containers or mono `"pcm16"` little-endian samples (`8000`–`48000` Hz).
3. **Evaluate or cancel:** Send `{{"action": "end"}}` to run the pipeline, or `{{"action": "reset"}}` at any time to clear the buffer and cancel an in-flight evaluation.
4. **Server progress events:** `ready` → `transcribing` → `retrieving` (`partial_text`) → `evaluating` → `completed` (`payload`) or `error` (`status_code`, `detail`).

## 6. Operational Limits and HTTP Status Codes

Audio uploads are capped at **5 MB** and **60 seconds**. CPU ASR execution is serialized (`concurrency=1`, **10 s** queue wait) with stage timeouts of **90 s** (ASR), **15 s** (RAG), and **30 s** (LLM).

| Status Code | Meaning | Client Action |
| :--- | :--- | :--- |
| `200 OK` | Request succeeded | Parse `EvaluationResponse`, `QueryResponse`, or `HealthResponse` |
| `400 Bad Request` | Empty file (`0 bytes`), unsupported extension, or invalid WS command | Verify file extension (`.wav`, `.mp3`, `.webm`, `.ogg`, `.m4a`) |
| `401 Unauthorized` | Missing or invalid `API_AUTH_TOKEN` (WS closes with `1008`) | Check `Authorization: Bearer`, `X-API-Key`, or `?token=` |
| `413 Content Too Large` | Payload exceeds **5 MB** (`5,242,880 bytes`) | Keep recordings under 60 s or use compressed `.webm`/`.mp3` |
| `422 Unprocessable Entity` | Duration > **60 s**, no speech detected, or invalid scenario/custom pair | Shorten audio or check `target_scenario` / `custom_scenario` rules |
| `429 Too Many Requests` | CPU ASR queue (`concurrency=1`) waited > **10 s** | Retry after a short backoff |
| `502 Bad Gateway` | Upstream Groq Cloud LLM error or invalid JSON | Verify `GROQ_API_KEY` and Groq quota |
| `503 Service Unavailable` | `faster-whisper` model is still initializing | Poll `GET /health` until `asr_ready` is `true` |
| `504 Gateway Timeout` | Stage exceeded timeout (**90 s** ASR, **15 s** RAG, **30 s** LLM) | Retry with shorter audio or verify Groq connectivity |
"""


def build_oracle_docs_html(project_name: str, version: str, description: str) -> str:
    """Render the dedicated Oracle-inspired API Consumption Documentation portal for /docs."""
    scenario_rows_html = []
    for member in TargetScenario:
        title = escape(SCENARIO_TITLES.get(member, member.value))
        desc = escape(SCENARIO_DESCRIPTIONS.get(member, ""))
        val = escape(member.value)
        scenario_rows_html.append(
            f"<tr><td><code>{val}</code></td><td><strong>{title}</strong></td><td>{desc}</td></tr>"
        )
    scenarios_tbody = "\n".join(scenario_rows_html)

    safe_name = escape(project_name)
    safe_ver = escape(version)
    safe_desc = escape(description)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{safe_name} · REST & WebSocket API Reference</title>
  <link rel="icon" type="image/svg+xml" href="{FAVICON_DATA_URI}" />
  <style>
    :root {{
      --ora-bg: #f4f6f8;
      --ora-surface: #ffffff;
      --ora-surface-alt: #f8fafc;
      --ora-border: #d5dce4;
      --ora-ink: #161b22;
      --ora-muted: #52606d;
      --ora-header: #1d2633;
      --ora-accent: #1b4b72;
      --ora-accent-soft: #eaf1f7;
      --ora-get: #1b4b72;
      --ora-post: #1f5c4d;
      --ora-ws: #475569;
      --ora-code-bg: #1e293b;
      --ora-code-ink: #f1f5f9;
      --ora-inline-bg: #edf1f5;
      --ora-inline-ink: #1e293b;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      max-width: 100%;
      overflow-x: hidden;
      background: var(--ora-bg);
      color: var(--ora-ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      line-height: 1.6;
      scroll-behavior: smooth;
    }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 20;
      background: var(--ora-header);
      border-bottom: 3px solid var(--ora-accent);
      color: #f8fafc;
      padding: 12px clamp(16px, 3vw, 32px);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .topbar-brand {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .topbar-brand h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.01em;
    }}
    .ver-pill {{
      background: rgba(255, 255, 255, 0.12);
      color: #e2e8f0;
      border: 1px solid rgba(255, 255, 255, 0.22);
      border-radius: 4px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 600;
    }}
    .topbar-nav {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .topbar-nav a {{
      color: #f8fafc;
      text-decoration: none;
      font-size: 13px;
      font-weight: 600;
      padding: 6px 12px;
      border-radius: 5px;
      border: 1px solid rgba(255, 255, 255, 0.2);
      background: rgba(255, 255, 255, 0.06);
    }}
    .topbar-nav a.primary-cta {{
      background: var(--ora-accent);
      border-color: var(--ora-accent);
    }}
    .topbar-nav a:hover {{
      background: rgba(255, 255, 255, 0.16);
    }}
    .layout {{
      max-width: 1280px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      gap: 24px;
      padding: 24px clamp(14px, 2.5vw, 28px) 48px;
      align-items: start;
    }}
    @media (max-width: 960px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        position: static !important;
      }}
    }}
    .sidebar {{
      position: sticky;
      top: 72px;
      background: var(--ora-surface);
      border: 1px solid var(--ora-border);
      border-radius: 6px;
      padding: 16px;
    }}
    .sidebar-title {{
      font-size: 11.5px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--ora-muted);
      margin: 0 0 10px 0;
    }}
    .sidebar ul {{
      list-style: none;
      margin: 0;
      padding: 0;
    }}
    .sidebar li {{
      margin-bottom: 4px;
    }}
    .sidebar a {{
      display: block;
      text-decoration: none;
      color: var(--ora-ink);
      font-size: 13.5px;
      padding: 6px 10px;
      border-radius: 5px;
      border-left: 3px solid transparent;
    }}
    .sidebar a:hover {{
      background: var(--ora-accent-soft);
      color: var(--ora-accent);
      border-left-color: var(--ora-accent);
    }}
    .sidebar .sub-link {{
      padding-left: 20px;
      font-size: 12.5px;
      color: var(--ora-muted);
    }}
    main {{
      min-width: 0;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }}
    .doc-section {{
      background: var(--ora-surface);
      border: 1px solid var(--ora-border);
      border-radius: 6px;
      padding: clamp(18px, 2.5vw, 28px);
      min-width: 0;
    }}
    .doc-section h2 {{
      margin: 0 0 12px 0;
      font-size: 20px;
      color: var(--ora-header);
      border-bottom: 1px solid var(--ora-border);
      padding-bottom: 8px;
    }}
    .doc-section h3 {{
      margin: 22px 0 8px 0;
      font-size: 15.5px;
      color: var(--ora-accent);
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .doc-section p, .doc-section li {{
      font-size: 14px;
      color: var(--ora-ink);
    }}
    .lead {{
      color: var(--ora-muted);
      margin-top: 0;
      font-size: 14.5px;
    }}
    code {{
      background: var(--ora-inline-bg);
      color: var(--ora-inline-ink);
      padding: 2px 6px;
      border-radius: 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12.5px;
    }}
    .code-wrap {{
      position: relative;
      margin: 10px 0 14px 0;
    }}
    .copy-btn {{
      position: absolute;
      top: 8px;
      right: 8px;
      background: rgba(255, 255, 255, 0.12);
      color: #f1f5f9;
      border: 1px solid rgba(255, 255, 255, 0.22);
      border-radius: 4px;
      padding: 4px 10px;
      font-size: 11.5px;
      font-weight: 600;
      cursor: pointer;
    }}
    .copy-btn:hover {{
      background: rgba(255, 255, 255, 0.24);
    }}
    pre {{
      margin: 0;
      background: var(--ora-code-bg);
      color: var(--ora-code-ink);
      border-radius: 6px;
      padding: 14px 16px;
      overflow-x: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12.5px;
      line-height: 1.5;
    }}
    pre code {{
      background: transparent;
      color: inherit;
      padding: 0;
    }}
    .table-wrap {{
      width: 100%;
      overflow-x: auto;
      margin: 12px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13.5px;
    }}
    th {{
      background: #edf1f5;
      color: var(--ora-header);
      font-weight: 600;
      text-align: left;
      padding: 9px 12px;
      border: 1px solid var(--ora-border);
    }}
    td {{
      padding: 9px 12px;
      border: 1px solid var(--ora-border);
      vertical-align: top;
    }}
    .method {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11.5px;
      font-weight: 700;
      color: #ffffff;
      letter-spacing: 0.03em;
    }}
    .method.get {{ background: var(--ora-get); }}
    .method.post {{ background: var(--ora-post); }}
    .method.ws {{ background: var(--ora-ws); }}
    .callout {{
      background: var(--ora-surface-alt);
      border-left: 4px solid var(--ora-accent);
      border-top: 1px solid var(--ora-border);
      border-right: 1px solid var(--ora-border);
      border-bottom: 1px solid var(--ora-border);
      border-radius: 4px;
      padding: 12px 16px;
      margin: 12px 0;
      font-size: 13.5px;
    }}
  </style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-brand">
      <h1>{safe_name} · REST &amp; WebSocket API Reference</h1>
      <span class="ver-pill">v{safe_ver}</span>
    </div>
    <nav class="topbar-nav">
      <a href="/ui" class="primary-cta">Consola de Pruebas (/ui)</a>
      <a href="/redoc">Referencia OpenAPI (/redoc)</a>
      <a href="/openapi.json">OpenAPI JSON</a>
    </nav>
  </header>

  <div class="layout">
    <aside class="sidebar">
      <div class="sidebar-title">Contenido de la API</div>
      <ul>
        <li><a href="#overview">1. Descripción y Rutas Base</a></li>
        <li><a href="#auth">2. Autenticación</a></li>
        <li><a href="#scenarios">3. Catálogo de Escenarios</a></li>
        <li><a href="#rest-endpoints">4. Servicios REST (Postman / cURL)</a></li>
        <li><a href="#ep-health" class="sub-link">GET /health</a></li>
        <li><a href="#ep-voice" class="sub-link">POST /api/v1/evaluate/voice</a></li>
        <li><a href="#ep-text" class="sub-link">POST /api/v1/evaluate/text</a></li>
        <li><a href="#ep-query" class="sub-link">POST /api/v1/query</a></li>
        <li><a href="#websocket">5. Protocolo WebSocket</a></li>
        <li><a href="#errors">6. Límites y Códigos HTTP</a></li>
      </ul>
    </aside>

    <main>
      <section class="doc-section" id="overview">
        <h2>1. Descripción y Rutas Base</h2>
        <p class="lead">{safe_desc}</p>
        <p>Esta guía documenta únicamente los contratos necesarios para consumir el servicio desde <strong>Postman</strong>, <strong>cURL</strong> o cualquier cliente HTTP/WebSocket externo. Para ejecutar pruebas interactivas desde el navegador (incluyendo micrófono), utilice la <a href="/ui" style="color:var(--ora-accent); font-weight:600;">Consola de Pruebas (/ui)</a>.</p>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>Recurso</th><th>Método / Ruta</th><th>Descripción</th></tr>
            </thead>
            <tbody>
              <tr><td>Estado del sistema</td><td><span class="method get">GET</span> <code>/health</code></td><td>Verifica disponibilidad de ChromaDB, Whisper ASR y Groq LLM.</td></tr>
              <tr><td>Evaluación de voz</td><td><span class="method post">POST</span> <code>/api/v1/evaluate/voice</code></td><td>Recibe audio (<code>multipart/form-data</code>) y devuelve transcripción + evaluación MCER/CEFR.</td></tr>
              <tr><td>Evaluación de texto</td><td><span class="method post">POST</span> <code>/api/v1/evaluate/text</code></td><td>Evalúa una respuesta técnica escrita en inglés (JSON).</td></tr>
              <tr><td>Búsqueda semántica</td><td><span class="method post">POST</span> <code>/api/v1/query</code></td><td>Consulta directa al corpus vectorial de ingeniería (sin ASR ni LLM).</td></tr>
              <tr><td>Práctica en tiempo real</td><td><span class="method ws">WS</span> <code>/api/v1/ws/practice</code></td><td>Canal WebSocket full-duplex con eventos de progreso y cancelación.</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="doc-section" id="auth">
        <h2>2. Autenticación (API_AUTH_TOKEN)</h2>
        <p>Todos los servicios expuestos operan bajo una política <strong>fail-closed</strong>: cualquier petición sin el token configurado en la variable <code>API_AUTH_TOKEN</code> (o su alias <code>API_TOKEN</code> en <code>.env</code>) será rechazada con <code>HTTP 401 Unauthorized</code>.</p>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>Cliente / Transporte</th><th>Ubicación del Token</th><th>Configuración Exacta</th></tr>
            </thead>
            <tbody>
              <tr><td><strong>Postman (Recomendado)</strong></td><td>Pestaña <em>Authorization</em> → <strong>Bearer Token</strong></td><td>Pegar el valor de <code>API_AUTH_TOKEN</code> en el campo <em>Token</em>.</td></tr>
              <tr><td><strong>Cabecera HTTP alternativa</strong></td><td>Header <code>X-API-Key</code></td><td><code>X-API-Key: &lt;API_AUTH_TOKEN&gt;</code></td></tr>
              <tr><td><strong>cURL / SDK HTTP</strong></td><td>Header <code>Authorization</code></td><td><code>Authorization: Bearer &lt;API_AUTH_TOKEN&gt;</code></td></tr>
              <tr><td><strong>WebSocket (Navegador / Cliente)</strong></td><td>Query Parameter <code>?token=</code></td><td><code>wss://&lt;host&gt;/api/v1/ws/practice?token=&lt;API_AUTH_TOKEN&gt;</code></td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="doc-section" id="scenarios">
        <h2>3. Catálogo de Escenarios (target_scenario y custom_scenario)</h2>
        <p>Los endpoints de evaluación soportan <strong>9 escenarios predefinidos</strong> de ingeniería de software y un modo personalizado explícito (<code>custom</code>).</p>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>Identificador (<code>target_scenario</code>)</th><th>Nombre</th><th>Contexto Evaluado</th></tr>
            </thead>
            <tbody>
              {scenarios_tbody}
            </tbody>
          </table>
        </div>
        <div class="callout">
          <strong>Reglas para modo personalizado (<code>target_scenario="custom"</code>):</strong><br/>
          1. Envíe <code>target_scenario</code> con el valor <code>"custom"</code>.<br/>
          2. Envíe el campo <code>custom_scenario</code> con una descripción de <strong>{CUSTOM_SCENARIO_MIN_LEN} a {CUSTOM_SCENARIO_MAX_LEN} caracteres imprimibles en una sola línea</strong>.<br/>
          3. Cuando utilice cualquiera de los 9 escenarios predefinidos, omita <code>custom_scenario</code> (o envíelo como <code>null</code>); de lo contrario recibirá <code>HTTP 422</code>.
        </div>
      </section>

      <section class="doc-section" id="rest-endpoints">
        <h2>4. Consumo de Servicios REST (Postman y cURL)</h2>

        <h3 id="ep-health"><span class="method get">GET</span> <code>/health</code> · Verificación de Disponibilidad</h3>
        <p>Devuelve el estado global (<code>ok</code> o <code>degraded</code>) y las banderas de precarga en memoria (<code>chroma_ready</code>, <code>asr_ready</code>, <code>llm_configured</code>).</p>
        <div class="code-wrap">
          <button class="copy-btn" onclick="copyCode(this)">Copiar</button>
          <pre><code>curl -X GET "&lt;BASE_URL&gt;/health" \
  -H "Authorization: Bearer &lt;API_AUTH_TOKEN&gt;"</code></pre>
        </div>

        <h3 id="ep-voice"><span class="method post">POST</span> <code>/api/v1/evaluate/voice</code> · Evaluación Multimodal de Audio</h3>
        <p><strong>Configuración en Postman:</strong> Método <code>POST</code> · Pestaña <strong>Body → form-data</strong> (no usar <em>raw</em> ni <em>binary</em>):</p>
        <ul>
          <li><code>audio_file</code> (tipo <strong>File</strong>, obligatorio): archivo <code>.wav</code>, <code>.mp3</code>, <code>.webm</code>, <code>.ogg</code> o <code>.m4a</code> (máximo <strong>5 MB</strong> y <strong>60 segundos</strong>).</li>
          <li><code>target_scenario</code> (tipo <strong>Text</strong>, obligatorio): uno de los 9 slugs predefinidos o <code>custom</code>.</li>
          <li><code>custom_scenario</code> (tipo <strong>Text</strong>, condicional): obligatorio solo si <code>target_scenario=custom</code>.</li>
          <li><code>expected_transcript</code> (tipo <strong>Text</strong>, opcional): texto leído de referencia para calcular <code>wer_score</code> y <code>cer_score</code>.</li>
        </ul>
        <div class="code-wrap">
          <button class="copy-btn" onclick="copyCode(this)">Copiar</button>
          <pre><code>curl -X POST "&lt;BASE_URL&gt;/api/v1/evaluate/voice" \
  -H "Authorization: Bearer &lt;API_AUTH_TOKEN&gt;" \
  -F "audio_file=@sample.wav" \
  -F "target_scenario=code_review"</code></pre>
        </div>

        <h3 id="ep-text"><span class="method post">POST</span> <code>/api/v1/evaluate/text</code> · Evaluación de Respuesta Escrita</h3>
        <p><strong>Configuración en Postman:</strong> Método <code>POST</code> · Pestaña <strong>Body → raw → JSON</strong> · Header <code>Content-Type: application/json</code>.</p>
        <div class="code-wrap">
          <button class="copy-btn" onclick="copyCode(this)">Copiar</button>
          <pre><code>curl -X POST "&lt;BASE_URL&gt;/api/v1/evaluate/text" \
  -H "Authorization: Bearer &lt;API_AUTH_TOKEN&gt;" \
  -H "Content-Type: application/json" \
  -d '{{
  "candidate_text": "We implemented horizontal database sharding and a Redis cache layer to mitigate write latency.",
  "target_scenario": "system_design_interview"
}}'</code></pre>
        </div>
        <p>Ejemplo con escenario personalizado (<code>custom</code>):</p>
        <div class="code-wrap">
          <button class="copy-btn" onclick="copyCode(this)">Copiar</button>
          <pre><code>{{
  "candidate_text": "We configured readiness probes and PodDisruptionBudgets to guarantee zero-downtime rolling updates.",
  "target_scenario": "custom",
  "custom_scenario": "Kubernetes zero-downtime deployment strategy review"
}}</code></pre>
        </div>

        <h3 id="ep-query"><span class="method post">POST</span> <code>/api/v1/query</code> · Búsqueda Semántica en Corpus RAG</h3>
        <p><strong>Configuración en Postman:</strong> Método <code>POST</code> · Pestaña <strong>Body → raw → JSON</strong>.</p>
        <div class="code-wrap">
          <button class="copy-btn" onclick="copyCode(this)">Copiar</button>
          <pre><code>curl -X POST "&lt;BASE_URL&gt;/api/v1/query" \
  -H "Authorization: Bearer &lt;API_AUTH_TOKEN&gt;" \
  -H "Content-Type: application/json" \
  -d '{{
  "query_text": "How does horizontal database sharding mitigate write bottlenecks?",
  "top_k": 3
}}'</code></pre>
        </div>
      </section>

      <section class="doc-section" id="websocket">
        <h2>5. Protocolo WebSocket en Tiempo Real (WS /api/v1/ws/practice)</h2>
        <p>Permite transmitir audio por fragmentos binarios y recibir eventos de avance por etapa en tiempo real. La conexión permanece abierta entre intervenciones.</p>
        <ol>
          <li><strong>Conexión:</strong> <code>wss://&lt;host&gt;/api/v1/ws/practice?token=&lt;API_AUTH_TOKEN&gt;</code>. Al aceptar, el servidor envía <code>{{"event": "ready"}}</code>.</li>
          <li><strong>Configuración (trama de texto JSON, opcional antes de cada intervención):</strong></li>
        </ol>
        <div class="code-wrap">
          <button class="copy-btn" onclick="copyCode(this)">Copiar</button>
          <pre><code>{{
  "action": "config",
  "target_scenario": "custom",
  "custom_scenario": "Distributed tracing and latency debugging",
  "format": "webm",
  "sample_rate": 16000,
  "expected_transcript": ""
}}</code></pre>
        </div>
        <ol start="3">
          <li><strong>Transmisión de audio:</strong> envíe uno o más <em>frames binarios</em> (límite acumulado: <strong>5 MB</strong> / <strong>60 s</strong>). <code>format</code> acepta contenedores <code>"webm"</code> o muestras crudas mono <code>"pcm16"</code> little-endian (<code>8000</code>–<code>48000</code> Hz).</li>
          <li><strong>Disparar o cancelar:</strong> envíe <code>{{"action": "end"}}</code> para evaluar, o <code>{{"action": "reset"}}</code> en cualquier momento para vaciar el buffer y cancelar una evaluación en curso.</li>
          <li><strong>Secuencia de eventos del servidor:</strong> <code>ready</code> → <code>transcribing</code> → <code>retrieving</code> (con <code>partial_text</code>) → <code>evaluating</code> → <code>completed</code> (con <code>payload</code>) o <code>error</code> (<code>status_code</code>, <code>detail</code>).</li>
        </ol>
      </section>

      <section class="doc-section" id="errors">
        <h2>6. Límites Operativos y Códigos de Respuesta HTTP</h2>
        <p>El motor procesa audio estrictamente en memoria (<code>io.BytesIO</code>) con límite de <strong>5 MB</strong> y <strong>60 segundos</strong>, concurrencia controlada en CPU (<code>ASR_MAX_CONCURRENCY=1</code>, cola máxima de <strong>10 s</strong>) y timeouts por etapa de <strong>90 s</strong> (ASR), <strong>15 s</strong> (RAG) y <strong>30 s</strong> (LLM).</p>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>Código</th><th>Significado</th><th>Acción del Cliente</th></tr>
            </thead>
            <tbody>
              <tr><td><code>200 OK</code></td><td>Evaluación o consulta exitosa</td><td>Procesar JSON de respuesta (<code>EvaluationResponse</code> / <code>QueryResponse</code>).</td></tr>
              <tr><td><code>400 Bad Request</code></td><td>Archivo vacío, extensión no soportada o acción WS desconocida</td><td>Verificar que el archivo tenga extensión válida (<code>.wav</code>, <code>.mp3</code>, <code>.webm</code>, <code>.ogg</code>, <code>.m4a</code>).</td></tr>
              <tr><td><code>401 Unauthorized</code></td><td>Falta <code>API_AUTH_TOKEN</code> o es inválido (WS cierra con <code>1008</code>)</td><td>Confirmar <code>Authorization: Bearer</code>, <code>X-API-Key</code> o <code>?token=</code>.</td></tr>
              <tr><td><code>413 Content Too Large</code></td><td>El audio excede <strong>5 MB</strong></td><td>Reducir duración (&lt; 60 s) o usar formato comprimido (<code>.webm</code>/<code>.mp3</code>).</td></tr>
              <tr><td><code>422 Unprocessable Entity</code></td><td>Duración &gt; <strong>60 s</strong>, sin voz detectable o escenario inválido</td><td>Revisar duración y validar <code>target_scenario</code> / <code>custom_scenario</code>.</td></tr>
              <tr><td><code>429 Too Many Requests</code></td><td>Cola de transcripción CPU saturada (&gt; <strong>10 s</strong>)</td><td>Reintentar tras unos segundos con backoff exponencial.</td></tr>
              <tr><td><code>502 Bad Gateway</code></td><td>Error de comunicación o JSON inválido desde Groq Cloud</td><td>Verificar cuota y clave <code>GROQ_API_KEY</code>.</td></tr>
              <tr><td><code>503 Service Unavailable</code></td><td>Motor <code>faster-whisper</code> no cargado aún</td><td>Consultar <code>GET /health</code> hasta que <code>asr_ready</code> sea <code>true</code>.</td></tr>
              <tr><td><code>504 Gateway Timeout</code></td><td>Timeout de etapa (<strong>90 s</strong> ASR / <strong>15 s</strong> RAG / <strong>30 s</strong> LLM)</td><td>Reintentar con audio más corto o revisar conectividad hacia Groq.</td></tr>
            </tbody>
          </table>
        </div>
      </section>
    </main>
  </div>

  <script>
    // Replace <BASE_URL> placeholder with the active deployment origin
    document.querySelectorAll('pre code').forEach((block) => {{
      block.textContent = block.textContent.replaceAll('<BASE_URL>', window.location.origin);
    }});

    function copyCode(btn) {{
      const code = btn.parentElement.querySelector('pre code').textContent;
      navigator.clipboard.writeText(code).then(() => {{
        const prev = btn.textContent;
        btn.textContent = 'Copiado';
        setTimeout(() => {{ btn.textContent = prev; }}, 1500);
      }});
    }}
  </script>
</body>
</html>"""


def build_redoc_html(project_name: str, openapi_url: str = "/openapi.json") -> str:
    """Render ReDoc 2.1.5 with Oracle-inspired neutral corporate light theme and favicon."""
    safe_name = escape(project_name)
    safe_openapi = escape(openapi_url)
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_name} · OpenAPI Reference (ReDoc)</title>
  <link rel="icon" type="image/svg+xml" href="{FAVICON_DATA_URI}" />
  <style>
    body {{
      margin: 0;
      padding: 0;
      background: #f4f6f8;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}
    .redoc-topbar {{
      background: #1d2633;
      border-bottom: 3px solid #1b4b72;
      color: #f8fafc;
      padding: 10px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .redoc-topbar strong {{
      font-size: 15px;
    }}
    .redoc-topbar a {{
      color: #f8fafc;
      text-decoration: none;
      font-size: 13px;
      font-weight: 600;
      padding: 5px 11px;
      border-radius: 5px;
      border: 1px solid rgba(255, 255, 255, 0.22);
      background: rgba(255, 255, 255, 0.08);
      margin-left: 8px;
    }}
    .redoc-topbar a:hover {{
      background: rgba(255, 255, 255, 0.18);
    }}
  </style>
</head>
<body>
  <div class="redoc-topbar">
    <strong>{safe_name} · Especificación OpenAPI (ReDoc)</strong>
    <div>
      <a href="/docs">Guía de Consumo (/docs)</a>
      <a href="/ui">Consola de Pruebas (/ui)</a>
    </div>
  </div>
  <div id="redoc-container"></div>
  <script src="https://cdn.jsdelivr.net/npm/redoc@2.1.5/bundles/redoc.standalone.js"></script>
  <script>
    Redoc.init("{safe_openapi}", {{
      scrollYOffset: 48,
      hideDownloadButton: false,
      theme: {{
        colors: {{
          primary: {{ main: "#1b4b72" }},
          text: {{ primary: "#161b22", secondary: "#52606d" }},
          http: {{
            get: "#1b4b72",
            post: "#1f5c4d",
            put: "#92400e",
            delete: "#991b1b"
          }}
        }},
        typography: {{
          fontSize: "14px",
          fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif",
          headings: {{
            fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif",
            fontWeight: "600"
          }},
          code: {{
            color: "#1e293b",
            backgroundColor: "#edf1f5"
          }}
        }},
        sidebar: {{
          backgroundColor: "#f4f6f8",
          textColor: "#161b22",
          activeTextColor: "#1b4b72"
        }},
        rightPanel: {{
          backgroundColor: "#1d2633",
          textColor: "#f1f5f9"
        }}
      }}
    }}, document.getElementById("redoc-container"));
  </script>
</body>
</html>"""
