"""OpenAPI / Swagger UI (/docs) Developer Integration Guide Builder.

Dynamically constructs the Markdown integration manual and tag descriptions from
the canonical TargetScenario catalog in app.schemas.request_schemas so Swagger UI
and ReDoc never drift when scenarios or limits change.
"""

from typing import Any, Dict, List

from app.schemas.request_schemas import (
    CUSTOM_SCENARIO_MAX_LEN,
    CUSTOM_SCENARIO_MIN_LEN,
    SCENARIO_DESCRIPTIONS,
    SCENARIO_TITLES,
    TargetScenario,
)

OPENAPI_TAGS: List[Dict[str, Any]] = [
    {
        "name": "System Information",
        "description": "Public root discovery index listing active endpoints, documentation, and interactive console URLs.",
    },
    {
        "name": "Health & Diagnostics",
        "description": "Subsystem readiness probe reporting in-memory status of ChromaDB (213,439 vectors), faster-whisper ASR, and Groq Cloud LLM.",
    },
    {
        "name": "Technical English Evaluation",
        "description": "Multimodal (voice audio -> ASR -> RAG -> Groq LLM) and text-only (RAG -> Groq LLM) CEFR technical English evaluation pipelines.",
    },
    {
        "name": "Semantic Retrieval",
        "description": "Direct cosine-similarity vector search over the ChromaDB software engineering corpus without invoking ASR or LLM.",
    },
    {
        "name": "Interactive Testing Console",
        "description": "Browser-based interactive testing playground (/ui) for microphone recording, WebSocket streaming, REST testing, and Postman/cURL generation.",
    },
]


def build_openapi_description(base_summary: str) -> str:
    """Build the comprehensive Markdown integration guide rendered at the top of /docs."""
    scenario_rows = []
    for member in TargetScenario:
        title = SCENARIO_TITLES.get(member, member.value)
        desc = SCENARIO_DESCRIPTIONS.get(member, "")
        scenario_rows.append(f"| `{member.value}` | **{title}** | {desc} |")
    scenario_table = "\n".join(scenario_rows)

    return f"""{base_summary}

---

## Interactive Browser Testing Console (`/ui`)

For an intuitive interface with **microphone recording, real-time WebSocket streaming, preset/custom scenario selection, and automatic Postman/cURL snippet generation**, open:

* **[Abrir Consola Interactiva de Pruebas (`/ui`)](/ui)**

---

## 1. Paso a Paso: Autenticación (`API_AUTH_TOKEN`)

Todas las rutas de negocio (`/health`, `/api/v1/*` y el canal WebSocket `/api/v1/ws/practice`) están protegidas bajo un modelo **fail-closed** usando la clave definida en el servidor como `API_AUTH_TOKEN` (o su alias `API_TOKEN` en el archivo `.env`).

| Cliente / Transporte | Cómo enviar la clave `API_AUTH_TOKEN` | Ejemplo |
| :--- | :--- | :--- |
| **Swagger UI (`/docs`)** | Haga clic en el botón **`Authorize`** (arriba a la derecha) y pegue su `API_AUTH_TOKEN` en `HTTPBearer` o `APIKeyHeader`. | `dG9rZW4t...` |
| **Postman / REST (Opción 1)** | En la pestaña **Authorization**, elija **Bearer Token** y pegue el valor de `API_AUTH_TOKEN`. | `Authorization: Bearer <API_AUTH_TOKEN>` |
| **Postman / REST (Opción 2)** | En la pestaña **Headers**, agregue la cabecera `X-API-Key`. | `X-API-Key: <API_AUTH_TOKEN>` |
| **WebSocket (`/api/v1/ws/practice`)** | Los navegadores no permiten cabeceras HTTP personalizadas en WebSockets; envíe el token por query param `?token=`. | `wss://<host>/api/v1/ws/practice?token=<API_AUTH_TOKEN>` |

---

## 2. Catálogo de Escenarios (`target_scenario` y `custom_scenario`)

El motor soporta **9 escenarios predefinidos de ingeniería de software** y un **modo personalizado (`custom`)**:

| Valor en API (`target_scenario`) | Nombre del Escenario | Contexto Evaluado por RAG + LLM |
| :--- | :--- | :--- |
{scenario_table}

### Cómo usar un escenario personalizado (`target_scenario="custom"`)
1. Envíe `target_scenario` con el valor `"custom"`.
2. Envíe el campo `custom_scenario` con una descripción de **{CUSTOM_SCENARIO_MIN_LEN} a {CUSTOM_SCENARIO_MAX_LEN} caracteres imprimibles en una sola línea** (por ejemplo: `"Kubernetes cluster DNS and CNI troubleshooting"`).
3. Si elige cualquiera de los 9 escenarios predefinidos, **no envíe `custom_scenario`** (o déjelo vacío/`null`); de lo contrario recibirá `HTTP 422`.

---

## 3. Guía de Consumo desde Postman y cURL

### A. Verificar estado del motor (`GET /health`)
```bash
curl -X GET "http://localhost:8000/health" \\
  -H "Authorization: Bearer <API_AUTH_TOKEN>"
```

### B. Evaluar respuesta en texto (`POST /api/v1/evaluate/text`)
* **En Postman:** Método `POST` · Body → `raw` → `JSON` · Cabecera `Content-Type: application/json`.
```bash
# Ejemplo con escenario predefinido:
curl -X POST "http://localhost:8000/api/v1/evaluate/text" \\
  -H "Authorization: Bearer <API_AUTH_TOKEN>" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "candidate_text": "We implemented horizontal database sharding and a Redis cache layer to mitigate write latency.",
    "target_scenario": "system_design_interview"
  }}'

# Ejemplo con escenario personalizado (custom):
curl -X POST "http://localhost:8000/api/v1/evaluate/text" \\
  -H "Authorization: Bearer <API_AUTH_TOKEN>" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "candidate_text": "We configured readiness probes and PodDisruptionBudgets to guarantee zero-downtime rolling updates.",
    "target_scenario": "custom",
    "custom_scenario": "Kubernetes zero-downtime deployment strategy review"
  }}'
```

### C. Evaluar respuesta de voz (`POST /api/v1/evaluate/voice`)
* **En Postman:** Método `POST` · Body → **`form-data`**:
  * `audio_file` (tipo **File**): archivo `.wav`, `.mp3`, `.webm`, `.ogg` o `.m4a` (máximo **5 MB** y **60 segundos**).
  * `target_scenario` (tipo **Text**): uno de los 9 escenarios o `custom`.
  * `custom_scenario` (tipo **Text**, opcional): obligatorio solo cuando `target_scenario=custom`.
  * `expected_transcript` (tipo **Text**, opcional): texto de referencia si desea calcular **WER** y **CER**.
```bash
curl -X POST "http://localhost:8000/api/v1/evaluate/voice" \\
  -H "Authorization: Bearer <API_AUTH_TOKEN>" \\
  -F "audio_file=@sample.wav" \\
  -F "target_scenario=code_review"
```

### D. Búsqueda semántica directa en el corpus (`POST /api/v1/query`)
```bash
curl -X POST "http://localhost:8000/api/v1/query" \\
  -H "Authorization: Bearer <API_AUTH_TOKEN>" \\
  -H "Content-Type: application/json" \\
  -d '{{"query_text": "How does horizontal database sharding mitigate write bottlenecks?", "top_k": 3}}'
```

---

## 4. Protocolo WebSocket en Tiempo Real (`WS /api/v1/ws/practice`)

> OpenAPI 3.x no lista rutas WebSocket en la tabla inferior. Puede probar este canal directamente desde **[/ui](/ui)** o con cualquier cliente WebSocket.

* **URL de conexión:** `ws://localhost:8000/api/v1/ws/practice?token=<API_AUTH_TOKEN>` (o `wss://` detrás de HTTPS/Cloudflare Tunnel).
* **Secuencia de mensajes (Cliente → Servidor):**
  1. **Configurar sesión (texto JSON, opcional antes de cada intervención):**
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
     *(`format` acepta `"webm"` para contenedores de audio o `"pcm16"` para muestras crudas mono de 16 bits little-endian entre 8000 y 48000 Hz).*
  2. **Enviar audio:** envíe uno o más *frames binarios* con los fragmentos de audio (hasta **5 MB** / **60 s** en total).
  3. **Finalizar locución y disparar evaluación:** envíe `{{"action": "end"}}`.
  4. **Cancelar evaluación en curso o limpiar buffer:** envíe `{{"action": "reset"}}` en cualquier momento.
* **Eventos de progreso (Servidor → Cliente):**
  * `{{"event": "ready"}}` (al conectar)
  * `{{"event": "transcribing"}}` (ejecutando `faster-whisper`)
  * `{{"event": "retrieving", "partial_text": "..."}}` (transcripción lista; consultando `ChromaDB`)
  * `{{"event": "evaluating"}}` (evaluando con `Groq Cloud`)
  * `{{"event": "completed", "payload": {{ ...EvaluationResponse... }}}}`
  * `{{"event": "error", "status_code": 422, "detail": "..."}}`

---

## 5. Límites Operativos, Concurrencia y Códigos de Error HTTP

* **Límites de audio:** máximo **5 MB** (`MAX_AUDIO_BYTES`) y **60 segundos** (`MAX_AUDIO_SECONDS`), validados en memoria (`io.BytesIO`) antes de iterar los segmentos de Whisper.
* **Concurrencia y Timeouts:** `ASR_MAX_CONCURRENCY = 1` en CPU con cola de espera máxima de **10 s** (`429`), y timeouts por etapa de **25 s** (ASR), **5 s** (RAG) y **30 s** (LLM).

| Código HTTP / WS | Causa | Acción recomendada |
| :--- | :--- | :--- |
| `400 Bad Request` | Archivo de audio vacío (`0 bytes`), extensión no soportada o comando WS desconocido. | Envíe un archivo `.wav`, `.mp3`, `.webm`, `.ogg` o `.m4a` con nombre y extensión válidos. |
| `401 Unauthorized` | Falta `API_AUTH_TOKEN` o no coincide con el configurado en el servidor (en WS cierra con código `1008`). | Verifique la cabecera `Authorization: Bearer <token>`, `X-API-Key` o `?token=`. |
| `413 Content Too Large` | El audio supera los **5 MB** (`5,242,880 bytes`). | Grabe intervenciones menores a 60 segundos o use un formato comprimido (`.webm`/`.mp3`). |
| `422 Unprocessable Entity` |Audio > **60 segundos**, audio sin habla inteligible, `target_scenario` desconocido o `custom_scenario` inválido. | Revise la duración del audio y que `custom_scenario` tenga entre 3 y 120 caracteres cuando `target_scenario="custom"`. |
| `429 Too Many Requests` | El motor ASR en CPU está ocupado y la espera en cola superó los **10 segundos**. | Reintente la petición tras unos segundos. |
| `502 Bad Gateway` | Error de comunicación o respuesta inválida desde Groq Cloud LLM. | Verifique `GROQ_API_KEY` y la cuota disponible en Groq Cloud. |
| `503 Service Unavailable` | El modelo `faster-whisper` aún no está inicializado en memoria. | Revise `GET /health` (`asr_ready`). |
| `504 Gateway Timeout` | Una etapa superó su límite de tiempo (**25 s** ASR, **5 s** RAG o **30 s** LLM). | Envíe un audio más corto o verifique la conectividad hacia Groq Cloud. |
"""
