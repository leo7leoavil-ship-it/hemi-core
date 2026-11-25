"""Zero-Dependency Interactive Web Testing Console & API Guide (/ui).

Replaces the legacy Gradio sandbox with a native HTML5/CSS/JS single-page testing
console served directly by FastAPI at `/ui`. Exercises the real REST and WebSocket
endpoints (`/health`, `/api/v1/evaluate/voice`, `/api/v1/evaluate/text`,
`/api/v1/query`, and `WS /api/v1/ws/practice`) using the user's `API_AUTH_TOKEN`.
"""

import json
from typing import Any, Dict, List

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.core.openapi_docs import FAVICON_DATA_URI
from app.schemas.request_schemas import (
    CUSTOM_SCENARIO_MAX_LEN,
    CUSTOM_SCENARIO_MIN_LEN,
    SCENARIO_DESCRIPTIONS,
    SCENARIO_TITLES,
    TargetScenario,
)

ui_router = APIRouter(tags=["Interactive Testing Console"])


def get_scenario_catalog() -> List[Dict[str, str]]:
    """Return the canonical scenario catalog derived from TargetScenario."""
    return [
        {
            "value": member.value,
            "title": SCENARIO_TITLES.get(member, member.value),
            "description": SCENARIO_DESCRIPTIONS.get(member, ""),
        }
        for member in TargetScenario
    ]


def build_testing_console_html() -> str:
    """Render the self-contained HTML/CSS/JS interactive console for /ui."""
    scenarios_json = json.dumps(get_scenario_catalog(), ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>hemi-core v1.0.0 · Consola Interactiva y Guía de API</title>
  <link rel="icon" type="image/svg+xml" href="{FAVICON_DATA_URI}" />
  <style>
    :root, [data-theme="light"] {{
      --bg: #f4f6f8;
      --panel: #ffffff;
      --panel-alt: #f8fafc;
      --border: #d5dce4;
      --text: #161b22;
      --label: #334155;
      --muted: #52606d;
      --header-bg: #1d2633;
      --header-text: #f8fafc;
      --header-sub: #cbd5e1;
      --primary: #1b4b72;
      --primary-hover: #143957;
      --accent-text: #1b4b72;
      --secondary-btn: #475569;
      --secondary-btn-hover: #334155;
      --success: #1f5c4d;
      --success-hover: #17473b;
      --success-text: #166534;
      --warning-text: #92400e;
      --danger: #991b1b;
      --danger-hover: #7f1d1d;
      --danger-text: #991b1b;
      --badge-bg: #e9eef4;
      --badge-text: #334155;
      --code-bg: #1e293b;
      --code-text: #f1f5f9;
      --inline-code-bg: #edf1f5;
      --inline-code-text: #1e293b;
    }}
    [data-theme="dark"] {{
      --bg: #0d1117;
      --panel: #161b22;
      --panel-alt: #0d1117;
      --border: #30363d;
      --text: #e6edf3;
      --label: #cbd5e1;
      --muted: #8b949e;
      --header-bg: #161b22;
      --header-text: #f0f6fc;
      --header-sub: #9eaab6;
      --primary: #2b5c8f;
      --primary-hover: #224a73;
      --accent-text: #79b8ff;
      --secondary-btn: #30363d;
      --secondary-btn-hover: #484f58;
      --success: #236452;
      --success-hover: #1b4f41;
      --success-text: #56d364;
      --warning-text: #e3b341;
      --danger: #8e2424;
      --danger-hover: #701c1c;
      --danger-text: #f85149;
      --badge-bg: #21262d;
      --badge-text: #c9d1d9;
      --code-bg: #0b0f14;
      --code-text: #e6edf3;
      --inline-code-bg: #21262d;
      --inline-code-text: #e6edf3;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      max-width: 100%;
      overflow-x: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      transition: background-color 0.2s ease, color 0.2s ease;
    }}
    code {{
      background: var(--inline-code-bg);
      color: var(--inline-code-text);
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.9em;
    }}
    .container {{
      width: 100%;
      max-width: 1180px;
      margin: 0 auto;
      padding: clamp(12px, 2.5vw, 24px);
    }}
    .header {{
      background: var(--header-bg);
      border: 1px solid var(--border);
      border-bottom: 3px solid var(--primary);
      border-radius: 8px;
      padding: clamp(16px, 2.5vw, 24px);
      margin-bottom: 18px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 14px;
    }}
    .header h1 {{
      margin: 0;
      color: var(--header-text);
      font-size: clamp(18px, 2.2vw, 23px);
      font-weight: 700;
      letter-spacing: -0.01em;
    }}
    .header p {{
      margin: 4px 0 0 0;
      color: var(--header-sub);
      font-size: clamp(12.5px, 1.4vw, 13.5px);
    }}
    .header-links {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }}
    .header-links a, .theme-toggle {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(255, 255, 255, 0.08);
      color: var(--header-text);
      text-decoration: none;
      padding: 7px 13px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      border: 1px solid rgba(255, 255, 255, 0.22);
      cursor: pointer;
    }}
    .header-links a:hover, .theme-toggle:hover {{
      background: rgba(255, 255, 255, 0.16);
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: clamp(14px, 2vw, 20px);
      margin-bottom: 18px;
      width: 100%;
      min-width: 0;
      max-width: 100%;
    }}
    .card-title {{
      font-size: 16px;
      font-weight: 700;
      margin: 0 0 10px 0;
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr));
      gap: 18px;
      width: 100%;
    }}
    .grid-2 > * {{
      min-width: 0;
      max-width: 100%;
    }}
    .stack-layout {{
      display: flex;
      flex-direction: column;
      gap: 18px;
      width: 100%;
    }}
    .feedback-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 340px), 1fr));
      gap: 12px;
      margin-top: 10px;
    }}
    .feedback-grid > * {{
      min-width: 0;
    }}
    label {{
      display: block;
      font-size: 13px;
      font-weight: 600;
      color: var(--label);
      margin-bottom: 6px;
    }}
    input[type="text"], input[type="password"], input[type="number"], select, textarea {{
      width: 100%;
      background: var(--panel-alt);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 10px 12px;
      border-radius: 6px;
      font-size: 14px;
      outline: none;
    }}
    input:focus, select:focus, textarea:focus {{
      border-color: var(--primary);
    }}
    .row-flex {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    button {{
      background: var(--primary);
      color: #fff;
      border: none;
      padding: 9px 15px;
      border-radius: 6px;
      font-size: 13.5px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s;
    }}
    button:hover {{ background: var(--primary-hover); }}
    button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
    button.btn-secondary {{ background: var(--secondary-btn); }}
    button.btn-secondary:hover {{ background: var(--secondary-btn-hover); }}
    button.btn-danger {{ background: var(--danger); }}
    button.btn-danger:hover {{ background: var(--danger-hover); }}
    button.btn-success {{ background: var(--success); }}
    button.btn-success:hover {{ background: var(--success-hover); }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      background: var(--badge-bg);
      color: var(--badge-text);
      border: 1px solid var(--border);
    }}
    .badge.ok {{ background: rgba(31, 92, 77, 0.14); color: var(--success-text); border: 1px solid var(--success); }}
    .badge.err {{ background: rgba(153, 27, 27, 0.14); color: var(--danger-text); border: 1px solid var(--danger); }}
    .badge.warn {{ background: rgba(180, 83, 9, 0.14); color: var(--warning-text); border: 1px solid #b45309; }}
    .help-text {{
      font-size: 12.5px;
      color: var(--muted);
      margin-top: 6px;
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 18px;
      flex-wrap: wrap;
    }}
    .tab-btn {{
      background: transparent;
      color: var(--muted);
      border: 1px solid transparent;
      border-bottom: none;
      padding: 10px 16px;
      border-radius: 8px 8px 0 0;
      font-weight: 600;
      cursor: pointer;
    }}
    .tab-btn.active {{
      background: var(--panel);
      color: var(--text);
      border-color: var(--border);
    }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 140px), 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .metric-box {{
      background: var(--panel-alt);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      text-align: center;
      min-width: 0;
    }}
    .metric-box .val {{
      font-size: 20px;
      font-weight: 700;
      color: var(--accent-text);
    }}
    .metric-box .lbl {{
      font-size: 11.5px;
      color: var(--muted);
    }}
    .feedback-item {{
      background: var(--panel-alt);
      border-left: 4px solid var(--primary);
      border-radius: 6px;
      padding: 12px 14px;
      margin-bottom: 10px;
      min-width: 0;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    .feedback-item h4 {{
      margin: 0 0 6px 0;
      font-size: 14px;
      color: var(--accent-text);
    }}
    .feedback-item p {{
      margin: 4px 0;
      font-size: 13px;
    }}
    pre {{
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      max-width: 100%;
      overflow-x: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
      font-size: 12.5px;
      color: var(--code-text);
      margin: 8px 0 0 0;
    }}
    .stepper {{
      display: flex;
      gap: 8px;
      margin: 12px 0;
      flex-wrap: wrap;
    }}
    .step {{
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      background: var(--panel-alt);
      border: 1px solid var(--border);
      color: var(--muted);
    }}
    .step.active {{
      border-color: var(--primary);
      background: rgba(27, 75, 114, 0.14);
      color: var(--accent-text);
    }}
    .step.done {{
      border-color: var(--success);
      background: rgba(31, 92, 77, 0.14);
      color: var(--success-text);
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h1>hemi-core v1.0.0 · Consola de Pruebas</h1>
        <p>Motor MaaS de Evaluación de Inglés Técnico (faster-whisper int8 + ChromaDB + Groq gpt-oss-120b)</p>
      </div>
      <div class="header-links">
        <button type="button" class="theme-toggle" id="themeToggleBtn">🌙 Tema Oscuro</button>
        <a href="/docs" target="_blank">📘 Documentación API (/docs)</a>
        <a href="/redoc" target="_blank">📄 ReDoc (/redoc)</a>
      </div>
    </div>

    <!-- PASO 1: AUTENTICACIÓN Y ESCENARIO -->
    <div class="grid-2">
      <div class="card">
        <div class="card-title">🔑 Paso 1 · Autenticación de la API (<code>API_AUTH_TOKEN</code>)</div>
        <label for="apiTokenInput">Clave de Acceso (<code>API_AUTH_TOKEN</code> / alias <code>API_TOKEN</code> en el archivo <code>.env</code> del servidor):</label>
        <div class="row-flex">
          <input type="password" id="apiTokenInput" placeholder="Pega aquí el valor de API_AUTH_TOKEN o API_TOKEN..." style="flex:1;" />
          <button type="button" class="btn-secondary" id="toggleTokenBtn">Mostrar</button>
          <button type="button" id="verifyHealthBtn">Verificar Token (/health)</button>
        </div>
        <div class="help-text">
          <strong>¿Qué clave debo ingresar?</strong> Es el token secreto definido en la variable <code>API_AUTH_TOKEN</code> (o <code>API_TOKEN</code>) del archivo <code>.env</code> de tu servidor. Se envía como cabecera <code>Authorization: Bearer &lt;API_AUTH_TOKEN&gt;</code> (o <code>X-API-Key</code>) en REST y como <code>?token=&lt;API_AUTH_TOKEN&gt;</code> en WebSocket.
        </div>
        <div class="row-flex" style="margin-top: 12px;" id="healthBadges">
          <span class="badge" id="badgeAuth">Auth: Sin verificar</span>
          <span class="badge" id="badgeChroma">ChromaDB: —</span>
          <span class="badge" id="badgeAsr">Whisper ASR: —</span>
          <span class="badge" id="badgeLlm">Groq LLM: —</span>
        </div>
      </div>

      <div class="card">
        <div class="card-title">🎯 Paso 2 · Selección de Escenario Técnico (<code>target_scenario</code>)</div>
        <label for="scenarioSelect">Escenario de práctica (9 predefinidos + modo personalizado):</label>
        <select id="scenarioSelect"></select>
        <div class="help-text" id="scenarioDescText"></div>

        <div id="customScenarioBox" style="display:none; margin-top: 12px;">
          <label for="customScenarioInput">
            Describe tu escenario personalizado (<code>custom_scenario</code>, {CUSTOM_SCENARIO_MIN_LEN}–{CUSTOM_SCENARIO_MAX_LEN} caracteres imprimibles):
            <span id="customCharCount" style="float:right; color:var(--accent-text);">0 / {CUSTOM_SCENARIO_MAX_LEN}</span>
          </label>
          <input
            type="text"
            id="customScenarioInput"
            maxlength="{CUSTOM_SCENARIO_MAX_LEN}"
            placeholder="Ej: Kubernetes cluster DNS debugging and CNI packet loss investigation"
          />
          <div class="help-text" id="customScenarioError" style="color:var(--danger-text); display:none;"></div>
        </div>
      </div>
    </div>

    <!-- PESTAÑAS DE PRUEBA Y GUÍA -->
    <div class="tabs">
      <button class="tab-btn active" data-tab="tab-voice">🎙️ 1. Voz REST (POST /api/v1/evaluate/voice)</button>
      <button class="tab-btn" data-tab="tab-ws">⚡ 2. Voz Streaming WebSocket (WS /api/v1/ws/practice)</button>
      <button class="tab-btn" data-tab="tab-text">📝 3. Texto REST (POST /api/v1/evaluate/text)</button>
      <button class="tab-btn" data-tab="tab-rag">🔍 4. Búsqueda RAG (POST /api/v1/query)</button>
      <button class="tab-btn" data-tab="tab-guide">📘 5. Guía Paso a Paso (Postman / cURL / API)</button>
    </div>

    <!-- TAB 1: VOICE REST -->
    <div class="tab-panel active" id="tab-voice">
      <div class="stack-layout">
        <div class="card">
          <div class="card-title">🎙️ Entrada de Audio (Límite: 5 MB / 60 segundos)</div>
          <p class="help-text" style="margin-top:0;">
            Puedes grabar directamente con tu micrófono en el navegador (se envía en memoria como <code>recording.webm</code>) o seleccionar un archivo <code>.wav</code>, <code>.mp3</code>, <code>.webm</code>, <code>.ogg</code> o <code>.m4a</code>.
          </p>
          <div class="row-flex" style="margin: 12px 0;">
            <button type="button" class="btn-success" id="recStartBtn">⏺️ Grabar Micrófono</button>
            <button type="button" class="btn-danger" id="recStopBtn" disabled>⏹️ Detener (<span id="recTimer">0.0s</span>)</button>
            <span id="recStatus" class="badge">Listo para grabar</span>
          </div>
          <audio id="audioPreview" controls style="width:100%; margin-bottom:12px; display:none;"></audio>

          <div class="grid-2">
            <div>
              <label for="audioFileInput">O sube un archivo de audio desde tu equipo:</label>
              <input type="file" id="audioFileInput" accept=".wav,.mp3,.webm,.ogg,.m4a,audio/*" />
            </div>
            <div>
              <label for="expectedTranscriptVoice">Texto de referencia opcional (<code>expected_transcript</code> para WER/CER):</label>
              <input type="text" id="expectedTranscriptVoice" placeholder="Opcional: texto leído para calcular Word Error Rate..." />
            </div>
          </div>

          <div class="row-flex" style="margin-top: 16px;">
            <button type="button" id="submitVoiceBtn">🚀 Evaluar Audio por REST (/api/v1/evaluate/voice)</button>
            <span id="voiceSpinner" class="help-text"></span>
          </div>

          <details style="margin-top: 14px;">
            <summary style="cursor:pointer; font-size:13px; color:var(--accent-text);">Ver comando cURL equivalente para terminal / Postman</summary>
            <pre id="curlVoicePreview"></pre>
          </details>
        </div>

        <div class="card" id="voiceResultCard">
          <div class="card-title">📊 Resultado de Evaluación Multimodal</div>
          <div id="voiceResultContainer">
            <p class="help-text">Graba o sube un audio y presiona <strong>Evaluar Audio por REST</strong> para ver la transcripción, nivel MCER/CEFR, métricas y sugerencias basadas en el corpus RAG.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: WEBSOCKET -->
    <div class="tab-panel" id="tab-ws">
      <div class="stack-layout">
        <div class="card">
          <div class="card-title">⚡ Streaming Full-Duplex (<code>WS /api/v1/ws/practice</code>)</div>
          <p class="help-text" style="margin-top:0;">
            Este canal abre una sesión WebSocket autenticada con <code>?token=&lt;API_AUTH_TOKEN&gt;</code>, envía la configuración del escenario (<code>{{"action":"config", ...}}</code>), transmite fragmentos binarios de audio, dispara la evaluación con <code>{{"action":"end"}}</code> y permite <strong>cancelar en vivo</strong> con <code>{{"action":"reset"}}</code>.
          </p>
          <label for="expectedTranscriptWs">Texto de referencia opcional (<code>expected_transcript</code>):</label>
          <input type="text" id="expectedTranscriptWs" placeholder="Opcional para WER/CER en WebSocket..." style="margin-bottom:12px;" />

          <div class="row-flex" style="margin-bottom: 12px;">
            <button type="button" id="wsConnectBtn">🔌 1. Conectar WebSocket</button>
            <button type="button" class="btn-success" id="wsRecBtn" disabled>⏺️ 2. Grabar y Transmitir</button>
            <button type="button" id="wsEndBtn" disabled>✅ 3. Finalizar y Evaluar (end)</button>
            <button type="button" class="btn-danger" id="wsResetBtn" disabled>🛑 Cancelar / Reset</button>
          </div>

          <div class="stepper" id="wsStepper">
            <span class="step" data-step="ready">1. ready</span>
            <span class="step" data-step="transcribing">2. transcribing (Whisper)</span>
            <span class="step" data-step="retrieving">3. retrieving (ChromaDB)</span>
            <span class="step" data-step="evaluating">4. evaluating (Groq LLM)</span>
            <span class="step" data-step="completed">5. completed</span>
          </div>

          <label>Bitácora de Tramas WebSocket en Vivo:</label>
          <pre id="wsLog" style="max-height: 180px; overflow-y: auto;">Sin conexión activa.</pre>
        </div>

        <div class="card" id="wsResultCard">
          <div class="card-title">📊 Resultado de Sesión WebSocket</div>
          <div id="wsResultContainer">
            <p class="help-text">Conecta el WebSocket, graba tu respuesta técnica y presiona <strong>Finalizar y Evaluar (end)</strong>.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: TEXT REST -->
    <div class="tab-panel" id="tab-text">
      <div class="stack-layout">
        <div class="card">
          <div class="card-title">📝 Evaluar Respuesta Escrita (<code>POST /api/v1/evaluate/text</code>)</div>
          <label for="candidateTextInput">Respuesta del candidato en inglés técnico (2–10,000 caracteres):</label>
          <textarea id="candidateTextInput" rows="4">We put a Redis cache in front of the primary database to make reads faster and we split the tables across servers so writes do not bottleneck.</textarea>
          <div class="row-flex" style="margin-top: 12px;">
            <button type="button" id="submitTextBtn">🚀 Evaluar Texto (/api/v1/evaluate/text)</button>
            <span id="textSpinner" class="help-text"></span>
          </div>
          <details style="margin-top: 14px;">
            <summary style="cursor:pointer; font-size:13px; color:var(--accent-text);">Ver comando cURL / Body JSON para Postman</summary>
            <pre id="curlTextPreview"></pre>
          </details>
        </div>

        <div class="card" id="textResultCard">
          <div class="card-title">📊 Resultado de Evaluación de Texto</div>
          <div id="textResultContainer">
            <p class="help-text">Presiona <strong>Evaluar Texto</strong> para ejecutar RAG + Groq LLM bajo el escenario seleccionado.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: RAG QUERY -->
    <div class="tab-panel" id="tab-rag">
      <div class="stack-layout">
        <div class="card">
          <div class="card-title">🔍 Búsqueda Semántica Directa en ChromaDB (<code>POST /api/v1/query</code>)</div>
          <div class="grid-2">
            <div>
              <label for="ragQueryInput">Consulta técnica en inglés (2–2,000 caracteres):</label>
              <input type="text" id="ragQueryInput" value="How does horizontal database sharding mitigate write bottlenecks?" />
            </div>
            <div>
              <label for="ragTopKInput">Número de fragmentos a recuperar (<code>top_k</code>, 1–20):</label>
              <input type="number" id="ragTopKInput" value="3" min="1" max="20" />
            </div>
          </div>
          <div style="margin-top: 12px;">
            <button type="button" id="submitRagBtn">🔍 Consultar Corpus Vectorial</button>
          </div>
        </div>

        <div class="card">
          <div class="card-title">📚 Fragmentos Recuperados (Similitud Coseno)</div>
          <div id="ragResultContainer">
            <p class="help-text">Consulta directamente los 213,439 vectores indexados con <code>all-MiniLM-L6-v2</code> sin invocar Whisper ni Groq.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 5: INTEGRATION GUIDE -->
    <div class="tab-panel" id="tab-guide">
      <div class="card">
        <div class="card-title">📘 Guía Completa para Consumir el Servicio desde Postman, cURL o Sistemas Externos</div>
        <p class="help-text" style="font-size:14px; color:var(--text);">
          Sigue estos pasos para integrar cualquier cliente externo con <strong>hemi-core</strong>:
        </p>
        <h4>1. Autenticación Obligatoria (Fail-Closed)</h4>
        <ul>
          <li>El servidor valida la variable <code>API_AUTH_TOKEN</code> (o su alias <code>API_TOKEN</code>) configurada en <code>.env</code>.</li>
          <li><strong>En Postman (REST):</strong> Abre la pestaña <em>Authorization</em> → selecciona <strong>Bearer Token</strong> → pega tu <code>API_AUTH_TOKEN</code> (o agrégala en <em>Headers</em> como <code>X-API-Key: &lt;API_AUTH_TOKEN&gt;</code>).</li>
          <li><strong>En WebSocket:</strong> Conecta a <code>ws(s)://&lt;host&gt;/api/v1/ws/practice?token=&lt;API_AUTH_TOKEN&gt;</code>. Si el token es inválido, el servidor cierra la conexión con código <code>1008 (Policy Violation)</code>.</li>
        </ul>

        <h4>2. Cómo enviar Audio desde Postman (<code>POST /api/v1/evaluate/voice</code>)</h4>
        <ul>
          <li>Selecciona método <strong>POST</strong> hacia <code>/api/v1/evaluate/voice</code>.</li>
          <li>En la pestaña <strong>Body</strong>, elige <strong>form-data</strong> (no elijas <em>raw</em> ni <em>binary</em>).</li>
          <li>Agrega las siguientes claves:
            <ul>
              <li><code>audio_file</code> (cambia el selector de la derecha de <em>Text</em> a <strong>File</strong>): selecciona un archivo <code>.wav</code>, <code>.mp3</code>, <code>.webm</code>, <code>.ogg</code> o <code>.m4a</code> de hasta <strong>5 MB</strong> y máximo <strong>60 segundos</strong>.</li>
              <li><code>target_scenario</code> (<em>Text</em>): cualquiera de los 9 escenarios predefinidos o <code>custom</code>.</li>
              <li><code>custom_scenario</code> (<em>Text</em>, opcional): <strong>obligatorio solo si <code>target_scenario=custom</code></strong> ({CUSTOM_SCENARIO_MIN_LEN}–{CUSTOM_SCENARIO_MAX_LEN} caracteres imprimibles en una sola línea).</li>
              <li><code>expected_transcript</code> (<em>Text</em>, opcional): texto de referencia para obtener <code>wer_score</code> y <code>cer_score</code>.</li>
            </ul>
          </li>
        </ul>

        <h4>3. Límites de Concurrencia, Timeouts y Códigos de Respuesta</h4>
        <ul>
          <li><strong>Semáforo ASR en CPU:</strong> procesa <code>1</code> transcripción a la vez con cola máxima de <strong>10 s</strong> (si se satura devuelve <code>HTTP 429 Too Many Requests</code>).</li>
          <li><strong>Timeouts por etapa:</strong> <strong>25 s</strong> para ASR, <strong>5 s</strong> para RAG y <strong>30 s</strong> para Groq LLM (si expira devuelve <code>HTTP 504 Gateway Timeout</code>).</li>
          <li><strong>Errores de validación:</strong> <code>400</code> archivo vacío o extensión inválida · <code>401</code> token inválido · <code>413</code> archivo &gt; 5 MB · <code>422</code> audio &gt; 60 s, sin voz detectable o escenario inválido · <code>502</code> error en Groq Cloud.</li>
        </ul>
      </div>
    </div>
  </div>

  <script>
    const SCENARIOS = {scenarios_json};
    const MIN_CUSTOM = {CUSTOM_SCENARIO_MIN_LEN};
    const MAX_CUSTOM = {CUSTOM_SCENARIO_MAX_LEN};

    // Corporate Light / Dark theme switcher
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    function applyTheme(theme) {{
      const t = theme === 'dark' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', t);
      localStorage.setItem('tt_ui_theme', t);
      themeToggleBtn.textContent = t === 'dark' ? '☀️ Tema Claro' : '🌙 Tema Oscuro';
    }}
    applyTheme(localStorage.getItem('tt_ui_theme') || 'light');
    themeToggleBtn.addEventListener('click', () => {{
      const current = document.documentElement.getAttribute('data-theme');
      applyTheme(current === 'dark' ? 'light' : 'dark');
    }});

    const tokenInput = document.getElementById('apiTokenInput');
    const scenarioSelect = document.getElementById('scenarioSelect');
    const scenarioDescText = document.getElementById('scenarioDescText');
    const customBox = document.getElementById('customScenarioBox');
    const customInput = document.getElementById('customScenarioInput');
    const customCharCount = document.getElementById('customCharCount');
    const customError = document.getElementById('customScenarioError');

    // Load token from query param or sessionStorage
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('token')) {{
      sessionStorage.setItem('tt_api_token', urlParams.get('token'));
    }}
    tokenInput.value = sessionStorage.getItem('tt_api_token') || '';
    tokenInput.addEventListener('input', () => {{
      sessionStorage.setItem('tt_api_token', tokenInput.value.trim());
      updateCurlPreviews();
    }});

    document.getElementById('toggleTokenBtn').addEventListener('click', (e) => {{
      const isPwd = tokenInput.type === 'password';
      tokenInput.type = isPwd ? 'text' : 'password';
      e.target.textContent = isPwd ? 'Ocultar' : 'Mostrar';
    }});

    // Populate scenarios from backend SSOT
    SCENARIOS.forEach((s) => {{
      const opt = document.createElement('option');
      opt.value = s.value;
      opt.textContent = `${{s.title}} (${{s.value}})`;
      scenarioSelect.appendChild(opt);
    }});

    function updateScenarioUI() {{
      const val = scenarioSelect.value;
      const found = SCENARIOS.find((s) => s.value === val);
      scenarioDescText.textContent = found ? found.description : '';
      customBox.style.display = val === 'custom' ? 'block' : 'none';
      validateCustomScenario();
      updateCurlPreviews();
    }}

    function validateCustomScenario() {{
      if (scenarioSelect.value !== 'custom') {{
        customError.style.display = 'none';
        return true;
      }}
      const txt = customInput.value.trim();
      customCharCount.textContent = `${{txt.length}} / ${{MAX_CUSTOM}}`;
      if (txt.length < MIN_CUSTOM) {{
        customError.textContent = `Ingresa al menos ${{MIN_CUSTOM}} caracteres para describir el escenario personalizado.`;
        customError.style.display = 'block';
        return false;
      }}
      customError.style.display = 'none';
      return true;
    }}

    scenarioSelect.addEventListener('change', updateScenarioUI);
    customInput.addEventListener('input', () => {{
      validateCustomScenario();
      updateCurlPreviews();
    }});
    document.getElementById('candidateTextInput').addEventListener('input', updateCurlPreviews);

    // Tabs logic
    document.querySelectorAll('.tab-btn').forEach((btn) => {{
      btn.addEventListener('click', () => {{
        document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.add('active');
      }});
    }});

    function getAuthHeaders() {{
      const tok = tokenInput.value.trim();
      return tok ? {{ 'Authorization': `Bearer ${{tok}}` }} : {{}};
    }}

    function updateCurlPreviews() {{
      const origin = window.location.origin;
      const tok = tokenInput.value.trim() || '<API_AUTH_TOKEN>';
      const scen = scenarioSelect.value;
      const custom = customInput.value.trim();

      let voiceCmd = `curl -X POST "${{origin}}/api/v1/evaluate/voice" \\\n  -H "Authorization: Bearer ${{tok}}" \\\n  -F "audio_file=@sample.wav" \\\n  -F "target_scenario=${{scen}}"`;
      if (scen === 'custom') {{
        voiceCmd += ` \\\n  -F "custom_scenario=${{custom || 'Kubernetes networking troubleshooting'}}"`;
      }}
      document.getElementById('curlVoicePreview').textContent = voiceCmd;

      const textPayload = {{
        candidate_text: document.getElementById('candidateTextInput').value.trim(),
        target_scenario: scen,
      }};
      if (scen === 'custom') {{
        textPayload.custom_scenario = custom || 'Kubernetes networking troubleshooting';
      }}
      document.getElementById('curlTextPreview').textContent =
        `curl -X POST "${{origin}}/api/v1/evaluate/text" \\\n  -H "Authorization: Bearer ${{tok}}" \\\n  -H "Content-Type: application/json" \\\n  -d '${{JSON.stringify(textPayload, null, 2)}}'`;
    }}

    updateScenarioUI();

    // Health check
    document.getElementById('verifyHealthBtn').addEventListener('click', async () => {{
      const bAuth = document.getElementById('badgeAuth');
      const bChroma = document.getElementById('badgeChroma');
      const bAsr = document.getElementById('badgeAsr');
      const bLlm = document.getElementById('badgeLlm');
      bAuth.className = 'badge warn';
      bAuth.textContent = 'Verificando...';
      try {{
        const res = await fetch('/health', {{ headers: getAuthHeaders() }});
        if (res.status === 401) {{
          bAuth.className = 'badge err';
          bAuth.textContent = 'Auth: 401 Token inválido o vacío';
          return;
        }}
        const data = await res.json();
        bAuth.className = 'badge ok';
        bAuth.textContent = `Auth: 200 OK (${{data.status}})`;
        bChroma.className = data.chroma_ready ? 'badge ok' : 'badge err';
        bChroma.textContent = `ChromaDB: ${{data.components?.chromadb_vector_store || data.chroma_ready}}`;
        bAsr.className = data.asr_ready ? 'badge ok' : 'badge err';
        bAsr.textContent = `Whisper: ${{data.asr_ready ? 'Listo' : 'No cargado'}}`;
        bLlm.className = data.llm_configured ? 'badge ok' : 'badge err';
        bLlm.textContent = `Groq LLM: ${{data.llm_configured ? 'Configurado' : 'Falta API Key'}}`;
      }} catch (err) {{
        bAuth.className = 'badge err';
        bAuth.textContent = `Error de red: ${{err.message}}`;
      }}
    }});

    async function parseApiJson(res) {{
      const rawText = await res.text();
      try {{
        return JSON.parse(rawText);
      }} catch (_) {{
        return {{
          status: 'error',
          http_status: res.status,
          detail: `HTTP ${{res.status}} (${{res.statusText || 'Respuesta no-JSON del servidor/proxy'}}): ${{rawText.replace(/<[^>]+>/g, ' ').replace(/\\s+/g, ' ').trim().slice(0, 280)}}`,
        }};
      }}
    }}

    function renderEvaluationResult(containerId, data, httpStatus) {{
      const el = document.getElementById(containerId);
      if (!data || !data.evaluation_core) {{
        const detailMsg = data && data.detail ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail, null, 2)) : JSON.stringify(data, null, 2);
        el.innerHTML = `
          <div class="feedback-item" style="border-left-color: var(--danger);">
            <h4 style="color: var(--danger-text);">⚠️ Error en la petición (HTTP ${{httpStatus || data?.http_status || 'Error'}})</h4>
            <p>${{detailMsg}}</p>
          </div>
        `;
        el.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
        return;
      }}
      const c = data.evaluation_core;
      const asr = data.asr_metrics;
      let html = `
        <div class="metrics-grid">
          <div class="metric-box"><div class="val">${{c.cefr_level}}</div><div class="lbl">Nivel MCER / CEFR</div></div>
          <div class="metric-box"><div class="val">${{c.overall_score}}</div><div class="lbl">Puntaje Global</div></div>
          <div class="metric-box"><div class="val">${{c.grammar_accuracy_score}}</div><div class="lbl">Gramática</div></div>
          <div class="metric-box"><div class="val">${{c.technical_precision_score}}</div><div class="lbl">Precisión Técnica</div></div>
          <div class="metric-box"><div class="val">${{c.lexical_density_ratio}}</div><div class="lbl">Densidad Léxica</div></div>
          <div class="metric-box"><div class="val">${{data.execution_time_ms}} ms</div><div class="lbl">Latencia Total</div></div>
        </div>
      `;
      if (asr) {{
        html += `
          <div class="card" style="background:var(--panel-alt); padding:12px; margin-bottom:12px;">
            <strong>🎙️ Transcripción Whisper (${{asr.speech_duration_seconds}}s · ${{asr.words_per_minute}} WPM · Confianza ${{(asr.confidence_score * 100).toFixed(1)}}%${{asr.wer_score !== null && asr.wer_score !== undefined ? ` · WER: ${{asr.wer_score}}` : ''}}):</strong>
            <p style="margin:6px 0 0 0; color:var(--text);">"${{asr.transcription}}"</p>
          </div>
        `;
      }}
      const terms = (c.detected_technical_terms || []).join(', ') || 'Ninguno detectado';
      html += `<p style="font-size:13px;"><strong>Términos técnicos detectados:</strong> <code>${{terms}}</code></p>`;

      if (data.feedback_details && data.feedback_details.length > 0) {{
        html += `<h4 style="margin:12px 0 8px 0;">Sugerencias Pedagógicas Grounded (RAG):</h4><div class="feedback-grid">`;
        data.feedback_details.forEach((f, idx) => {{
          html += `
            <div class="feedback-item">
              <h4>${{idx + 1}}. [${{f.type}}]</h4>
              <p><strong>Original:</strong> <code>${{f.original_phrase}}</code></p>
              <p><strong>Sugerido:</strong> <strong style="color:var(--success-text);">${{f.suggested_phrase}}</strong></p>
              <p><strong>Explicación:</strong> ${{f.explanation}}</p>
              <p class="help-text"><strong>Fuente RAG:</strong> ${{f.rag_source_context}}</p>
            </div>
          `;
        }});
        html += `</div>`;
      }}
      html += `<details style="margin-top:10px;"><summary style="cursor:pointer; font-size:13px; color:var(--accent-text);">Ver JSON completo</summary><pre>${{JSON.stringify(data, null, 2)}}</pre></details>`;
      el.innerHTML = html;
      el.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
    }}

    // Microphone Recording for REST Voice Tab
    let restRecorder = null;
    let restChunks = [];
    let recordedBlob = null;
    let recInterval = null;
    let recStartTime = 0;

    document.getElementById('recStartBtn').addEventListener('click', async () => {{
      try {{
        const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
        restChunks = [];
        recordedBlob = null;
        restRecorder = new MediaRecorder(stream);
        restRecorder.ondataavailable = (e) => {{
          if (e.data && e.data.size > 0) restChunks.push(e.data);
        }};
        restRecorder.onstop = () => {{
          stream.getTracks().forEach((t) => t.stop());
          clearInterval(recInterval);
          recordedBlob = new Blob(restChunks, {{ type: 'audio/webm' }});
          const url = URL.createObjectURL(recordedBlob);
          const preview = document.getElementById('audioPreview');
          preview.src = url;
          preview.style.display = 'block';
          document.getElementById('recStatus').className = 'badge ok';
          document.getElementById('recStatus').textContent = `Grabado (${{(recordedBlob.size / 1024).toFixed(1)}} KB)`;
        }};
        restRecorder.start(250);
        recStartTime = performance.now();
        document.getElementById('recStartBtn').disabled = true;
        document.getElementById('recStopBtn').disabled = false;
        document.getElementById('recStatus').className = 'badge warn';
        document.getElementById('recStatus').textContent = 'Grabando... (máx 60s)';
        recInterval = setInterval(() => {{
          const elapsed = (performance.now() - recStartTime) / 1000;
          document.getElementById('recTimer').textContent = `${{elapsed.toFixed(1)}}s`;
          if (elapsed >= 59.5 && restRecorder && restRecorder.state === 'recording') {{
            restRecorder.stop();
            document.getElementById('recStartBtn').disabled = false;
            document.getElementById('recStopBtn').disabled = true;
          }}
        }}, 100);
      }} catch (err) {{
        alert('No se pudo acceder al micrófono: ' + err.message);
      }}
    }});

    document.getElementById('recStopBtn').addEventListener('click', () => {{
      if (restRecorder && restRecorder.state === 'recording') {{
        restRecorder.stop();
      }}
      document.getElementById('recStartBtn').disabled = false;
      document.getElementById('recStopBtn').disabled = true;
    }});

    // Submit Voice REST
    document.getElementById('submitVoiceBtn').addEventListener('click', async () => {{
      if (!validateCustomScenario()) return;
      const fileInput = document.getElementById('audioFileInput');
      const file = fileInput.files[0];
      if (!file && !recordedBlob) {{
        alert('Graba un audio con el micrófono o selecciona un archivo primero.');
        return;
      }}
      const formData = new FormData();
      if (file) {{
        formData.append('audio_file', file, file.name);
      }} else {{
        formData.append('audio_file', recordedBlob, 'recording.webm');
      }}
      formData.append('target_scenario', scenarioSelect.value);
      if (scenarioSelect.value === 'custom') {{
        formData.append('custom_scenario', customInput.value.trim());
      }}
      const exp = document.getElementById('expectedTranscriptVoice').value.trim();
      if (exp) formData.append('expected_transcript', exp);

      const spinner = document.getElementById('voiceSpinner');
      spinner.textContent = 'Procesando audio (Whisper -> ChromaDB -> Groq)...';
      try {{
        const res = await fetch('/api/v1/evaluate/voice', {{
          method: 'POST',
          headers: getAuthHeaders(),
          body: formData,
        }});
        const data = await parseApiJson(res);
        spinner.textContent = `HTTP ${{res.status}}`;
        renderEvaluationResult('voiceResultContainer', data, res.status);
      }} catch (err) {{
        spinner.textContent = 'Error de red';
        document.getElementById('voiceResultContainer').innerHTML = `<pre>${{err.message}}</pre>`;
      }}
    }});

    // Submit Text REST
    document.getElementById('submitTextBtn').addEventListener('click', async () => {{
      if (!validateCustomScenario()) return;
      const payload = {{
        candidate_text: document.getElementById('candidateTextInput').value.trim(),
        target_scenario: scenarioSelect.value,
      }};
      if (scenarioSelect.value === 'custom') {{
        payload.custom_scenario = customInput.value.trim();
      }}
      const spinner = document.getElementById('textSpinner');
      spinner.textContent = 'Evaluando con ChromaDB + Groq...';
      try {{
        const res = await fetch('/api/v1/evaluate/text', {{
          method: 'POST',
          headers: {{ ...getAuthHeaders(), 'Content-Type': 'application/json' }},
          body: JSON.stringify(payload),
        }});
        const data = await parseApiJson(res);
        spinner.textContent = `HTTP ${{res.status}}`;
        renderEvaluationResult('textResultContainer', data, res.status);
      }} catch (err) {{
        spinner.textContent = 'Error de red';
        document.getElementById('textResultContainer').innerHTML = `<pre>${{err.message}}</pre>`;
      }}
    }});

    // Submit RAG Query
    document.getElementById('submitRagBtn').addEventListener('click', async () => {{
      const payload = {{
        query_text: document.getElementById('ragQueryInput').value.trim(),
        top_k: parseInt(document.getElementById('ragTopKInput').value || '3', 10),
      }};
      const container = document.getElementById('ragResultContainer');
      container.innerHTML = '<p class="help-text">Consultando ChromaDB...</p>';
      try {{
        const res = await fetch('/api/v1/query', {{
          method: 'POST',
          headers: {{ ...getAuthHeaders(), 'Content-Type': 'application/json' }},
          body: JSON.stringify(payload),
        }});
        const data = await res.json();
        if (!res.ok) {{
          container.innerHTML = `<div class="badge err">HTTP ${{res.status}}</div><pre>${{JSON.stringify(data, null, 2)}}</pre>`;
          return;
        }}
        let html = '';
        (data.chunks || []).forEach((c, i) => {{
          html += `
            <div class="feedback-item">
              <h4>#${{i + 1}} · Fuente: ${{c.source}} (Distancia Coseno: ${{c.distance}})</h4>
              <p style="white-space: pre-wrap;">${{c.document}}</p>
            </div>
          `;
        }});
        container.innerHTML = html || '<p class="help-text">Sin resultados.</p>';
      }} catch (err) {{
        container.innerHTML = `<pre>${{err.message}}</pre>`;
      }}
    }});

    // WebSocket Practice Client
    let ws = null;
    let wsRecorder = null;
    let wsStream = null;
    const wsLog = document.getElementById('wsLog');

    function logWs(dir, msg) {{
      const ts = new Date().toLocaleTimeString();
      wsLog.textContent += `\n[${{ts}}] ${{dir}} ${{typeof msg === 'string' ? msg : JSON.stringify(msg)}}`;
      wsLog.scrollTop = wsLog.scrollHeight;
    }}

    function setWsStep(activeStep) {{
      const order = ['ready', 'transcribing', 'retrieving', 'evaluating', 'completed'];
      const idx = order.indexOf(activeStep);
      document.querySelectorAll('#wsStepper .step').forEach((el) => {{
        const sIdx = order.indexOf(el.dataset.step);
        el.className = 'step' + (sIdx === idx ? ' active' : (sIdx < idx && idx !== -1 ? ' done' : ''));
      }});
    }}

    function sendWsConfig() {{
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const cfg = {{
        action: 'config',
        target_scenario: scenarioSelect.value,
        format: 'webm',
        expected_transcript: document.getElementById('expectedTranscriptWs').value.trim() || null,
      }};
      if (scenarioSelect.value === 'custom') {{
        cfg.custom_scenario = customInput.value.trim();
      }}
      ws.send(JSON.stringify(cfg));
      logWs('➡️ CLIENT:', cfg);
    }}

    document.getElementById('wsConnectBtn').addEventListener('click', () => {{
      if (!validateCustomScenario()) return;
      if (ws && ws.readyState === WebSocket.OPEN) {{
        ws.close();
        return;
      }}
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const tok = encodeURIComponent(tokenInput.value.trim());
      const url = `${{proto}}//${{window.location.host}}/api/v1/ws/practice?token=${{tok}}`;
      wsLog.textContent = `Conectando a ${{proto}}//${{window.location.host}}/api/v1/ws/practice...`;
      ws = new WebSocket(url);

      ws.onopen = () => {{
        document.getElementById('wsConnectBtn').textContent = '🔌 Desconectar WS';
        document.getElementById('wsRecBtn').disabled = false;
        document.getElementById('wsResetBtn').disabled = false;
      }};

      ws.onmessage = (ev) => {{
        try {{
          const data = JSON.parse(ev.data);
          logWs('⬅️ SERVER:', data.event === 'completed' ? {{ event: 'completed', execution_time_ms: data.payload?.execution_time_ms }} : data);
          if (data.event === 'ready') {{
            setWsStep('ready');
            sendWsConfig();
          }} else if (data.event === 'transcribing') {{
            setWsStep('transcribing');
          }} else if (data.event === 'retrieving') {{
            setWsStep('retrieving');
          }} else if (data.event === 'evaluating') {{
            setWsStep('evaluating');
          }} else if (data.event === 'completed') {{
            setWsStep('completed');
            renderEvaluationResult('wsResultContainer', data.payload);
          }} else if (data.event === 'error') {{
            document.getElementById('wsResultContainer').innerHTML =
              `<div class="badge err">Error WS ${{data.status_code}}</div><pre>${{data.detail}}</pre>`;
          }}
        }} catch (e) {{
          logWs('⬅️ SERVER:', ev.data);
        }}
      }};

      ws.onclose = (ev) => {{
        document.getElementById('wsConnectBtn').textContent = '🔌 1. Conectar WebSocket';
        document.getElementById('wsRecBtn').disabled = true;
        document.getElementById('wsEndBtn').disabled = true;
        document.getElementById('wsResetBtn').disabled = true;
        if (ev.code === 1008 || ev.code === 1006) {{
          logWs('❌ CLOSED:', `Código ${{ev.code}} (Verifica que tu API_AUTH_TOKEN en el Paso 1 sea válido).`);
        }} else {{
          logWs('ℹ️ CLOSED:', `Código ${{ev.code}}`);
        }}
      }};
    }});

    document.getElementById('wsRecBtn').addEventListener('click', async () => {{
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      if (!validateCustomScenario()) return;
      sendWsConfig();
      try {{
        wsStream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
        wsRecorder = new MediaRecorder(wsStream);
        wsRecorder.ondataavailable = async (e) => {{
          if (e.data && e.data.size > 0 && ws && ws.readyState === WebSocket.OPEN) {{
            const buf = await e.data.arrayBuffer();
            ws.send(buf);
            logWs('➡️ BINARY CHUNK:', `${{buf.byteLength}} bytes`);
          }}
        }};
        wsRecorder.start(500);
        document.getElementById('wsRecBtn').disabled = true;
        document.getElementById('wsEndBtn').disabled = false;
        logWs('🎙️ MIC:', 'Transmitiendo fragmentos de audio cada 500 ms...');
      }} catch (err) {{
        alert('Error de micrófono: ' + err.message);
      }}
    }});

    document.getElementById('wsEndBtn').addEventListener('click', () => {{
      if (wsRecorder && wsRecorder.state === 'recording') {{
        wsRecorder.onstop = () => {{
          if (wsStream) wsStream.getTracks().forEach((t) => t.stop());
          if (ws && ws.readyState === WebSocket.OPEN) {{
            ws.send(JSON.stringify({{ action: 'end' }}));
            logWs('➡️ CLIENT:', {{ action: 'end' }});
          }}
        }};
        wsRecorder.stop();
      }} else if (ws && ws.readyState === WebSocket.OPEN) {{
        ws.send(JSON.stringify({{ action: 'end' }}));
        logWs('➡️ CLIENT:', {{ action: 'end' }});
      }}
      document.getElementById('wsRecBtn').disabled = false;
      document.getElementById('wsEndBtn').disabled = true;
    }});

    document.getElementById('wsResetBtn').addEventListener('click', () => {{
      if (wsRecorder && wsRecorder.state === 'recording') {{
        wsRecorder.stop();
        if (wsStream) wsStream.getTracks().forEach((t) => t.stop());
      }}
      if (ws && ws.readyState === WebSocket.OPEN) {{
        ws.send(JSON.stringify({{ action: 'reset' }}));
        logWs('🛑 CLIENT:', {{ action: 'reset' }});
      }}
      document.getElementById('wsRecBtn').disabled = false;
      document.getElementById('wsEndBtn').disabled = true;
      setWsStep('ready');
    }});
  </script>
</body>
</html>"""


@ui_router.get(
    "/ui",
    response_class=HTMLResponse,
    summary="Interactive Browser Testing Console & API Guide",
    description="Zero-dependency HTML/JS console for testing voice (REST & WebSocket), text, and RAG endpoints with API_AUTH_TOKEN onboarding.",
)
@ui_router.get("/ui/", response_class=HTMLResponse, include_in_schema=False)
async def serve_testing_console() -> HTMLResponse:
    """Serve the interactive testing console HTML shell."""
    return HTMLResponse(content=build_testing_console_html())


def create_gradio_demo() -> str:
    """Backward-compatible helper returning the HTML testing console."""
    return build_testing_console_html()
