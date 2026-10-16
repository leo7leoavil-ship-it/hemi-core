# ESPECIFICACIÓN TÉCNICA FORMAL Y BLUEPRINT DE PRODUCCIÓN: MOTOR MAAS DE EVALUACIÓN DE INGLÉS TÉCNICO

**Proyecto de Tesis:** Asistente de Inteligencia Artificial Especializado en Inglés Técnico para Ingenieros del Área de Tecnología

**Institución:** Universidad Autónoma de Tlaxcala — Unidad Académica Multidisciplinaria Campus Calpulalpan

**Programa Académico:** Licenciatura en Ingeniería en Computación

**Autor:** Jesus Leonardo Beltrán Avila

**Rol Operativo:** Arquitecto de Soluciones de IA & Mentor de Tesis Senior (`/tesis`)

**Estado del Repositorio:** Preparación para despliegue productivo y desacoplamiento Git/MLOps

**Fecha de Emisión Técnica:** Octubre 2026

**Periodo de Defensa de Grado:** Diciembre 2026 – Enero 2027

---

## 1. Alcance Técnico y Justificación del Pivote Arquitectónico

### 1.1. Delimitación del Paradigma Model-as-a-Service (MaaS)

En consonancia con la reestructuración estratégica aprobada para garantizar el rigor metodológico y la viabilidad de culminación en tiempo y forma, el proyecto elimina cualquier dependencia de interfaces gráficas monolíticas de usuario final (descartando el desarrollo frontend en React, Next.js y el maquetado del agente Hemi, los cuales quedan formalmente catalogados como la línea primaria de Trabajo Futuro).

El entregable de ingeniería se consolida exclusivamente como un **Motor de Inferencia y Evaluación Técnica Asíncrono (Model-as-a-Service - MaaS)**. La solución expone capacidades de procesamiento digital de señales acústicas, recuperación semántica aumentada (RAG) y razonamiento analítico mediante protocolos de red abiertos y estandarizados:

* **API RESTful Asíncrona:** Basada en el framework de alto rendimiento FastAPI, orientada a peticiones transaccionales de evaluación fonética, análisis de texto e introspección de salud operativa del sistema.


* **Canales Bidireccionales Full-Duplex (WebSockets):** Diseñados para la ingesta continua de tramas binarias de audio y la emisión reactiva de estados de inferencia en tiempo real.


* **Contratos Estrictos de Datos:** Validación unificada bajo modelos Pydantic v2 para garantizar la integridad, tipado estático y serialización determinista de las respuestas en formato JSON.



```
+----------------------------------------------------------------------------------------------------+
|                                    ARQUITECTURA DISTRIBUIDA MAAS                                   |
+----------------------------------------------------------------------------------------------------+
| [Cliente / OpenAPI / Automated Benchmark Suite / Postman]                                          |
|                                     │                                                              |
|                                     │ HTTPS (REST) / WSS (WebSockets)                              |
|                                     ▼                                                              |
| ┌────────────────────────────────────────────────────────────────────────────────────────────────┐ │
| │ NODO DE CÓMPUTO Y ORQUESTACIÓN (Hugging Face Spaces - Docker Runtime - Puerto 7860)            │ │
| │                                                                                                │ │
| │  ┌──────────────────────────────────────────────────────────────────────────────────────────┐  │ │
| │  │ CAPA DE INGESTA Y CONTROL ASÍNCRONO (FastAPI + Uvicorn Async Event Loop)                 │  │ │
| │  │ • Sonda de Liveness/Readiness: GET /health                                               │  │ │
| │  │ • Búsqueda Semántica Vectorial: POST /api/v1/query                                       │  │ │
| │  │ • Pipeline E2E de Voz: POST /api/v1/evaluate/voice                                       │  │ │
| │  │ • Streaming de Audio en Tiempo Real: WS /api/v1/ws/practice                              │  │ │
| │  └──────────────────────────────┬────────────────────────────────────┬──────────────────────┘  │ │
| │                                 │                                    │                         │ │
| │                                 ▼ (asyncio.to_thread)                ▼ (asyncio.to_thread)     │ │
| │  ┌──────────────────────────────────────────────┐ ┌─────────────────────────────────────────┐  │ │
| │  │ SUBSISTEMA ASR LOCAL                         │ │ SUBSISTEMA RAG LOCAL                    │  │ │
| │  │ • Faster-Whisper (Large-v3-Turbo)            │ │ • ChromaDB Engine (tech_tutor_corpus)   │  │ │
| │  │ • Inferencia CTranslate2 (CPU / int8)        │ │ • 213,439 Vectores Normalizados         │  │ │
| │  │ • Ingesta de Audio en memoria (io.BytesIO)   │ │ • Modelo all-MiniLM-L6-v2 (384d en CPU) │  │ │
| │  │ • Prompt de Contexto Léxico Especializado    │ │ • Sanitización Regex Post-Retrieval     │  │ │
| │  └──────────────────────────────┬───────────────┘ └──────────────────┬──────────────────────┘  │ │
| └─────────────────────────────────┼────────────────────────────────────┼─────────────────────────┘ │
|                                   │ Transcripción                      │ Contexto Ground Truth     |
|                                   └─────────────────┬──────────────────┘                           |
|                                                     ▼                                              |
|                           ┌───────────────────────────────────────────────────┐                    |
|                           │ INVOCACIÓN I/O ASÍNCRONA (AsyncOpenAI Client)     │                    |
|                           └─────────────────────────┬─────────────────────────┘                    |
|                                                     │ HTTPS (Puerto 443 / Bearer Token)            |
+-----------------------------------------------------┼----------------------------------------------+
                                                      ▼
                      ┌──────────────────────────────────────────────────────────────┐
                      │ INFRAESTRUCTURA DE RAZONAMIENTO EXTERNA                      │
                      │ Groq Cloud LPU Processing Units (api.groq.com/openai/v1)     │
                      │ • Motor LLM: openai/gpt-oss-120b                             │
                      │ • Decodificación MoE ultrarrápida (~500 tokens/segundo)       │
                      │ • Evaluación Rúbrica MCER y Structured JSON Output           │
                      └──────────────────────────────────────────────────────────────┘

```

### 1.2. Transición Metodológica del Motor de Razonamiento hacia `openai/gpt-oss-120b`

En las versiones preliminares del protocolo y de los informes técnicos, se contemplaba la utilización de modelos de la familia Llama 3 (específicamente `llama-3.3-70b-versatile`). Sin embargo, para la versión definitiva de producción y defensa de tesis, el núcleo de razonamiento se actualiza formalmente a **`openai/gpt-oss-120b`** servido a través de la infraestructura de aceleración por LPUs de Groq Cloud:

1. **Capacidad de Evaluación como Juez Automatizado (*LLM-as-a-Judge*):** Con una arquitectura basada en *Mixture-of-Experts* (MoE) que activa aproximadamente 5.1 mil millones de parámetros por *forward pass*, `gpt-oss-120b` ofrece una densidad de comprensión contextual y metalingüística superior. Esto permite contrastar de manera simultánea las directrices pedagógicas del Marco Común Europeo de Referencia para las Lenguas (MCER en bandas B2 y C1) con el contexto técnico recuperado de la base vectorial.


2. **Latencia Extremadamente Reducida en LPUs:** Groq procesa `gpt-oss-120b` a tasas cercanas a 500 tokens por segundo, lo que mantiene el tiempo de generación del dictamen estructurado JSON por debajo de los 600 milisegundos, preservando el presupuesto de latencia interactiva de la API.


3. **Alineación Determinista de Esquemas:** El modelo cuenta con soporte nativo para *Structured Outputs*, mitigando el riesgo de errores de sintaxis JSON o desbordamiento de campos no tipados al interactuar con Pydantic v2.



---

## 2. Blueprint y Arquitectura del Repositorio de Producción

Para garantizar la aplicación estricta de *Clean Architecture* e inversión de dependencias en Python 3.11+, el repositorio `tech-tutor-engine` mantiene la siguiente topología de archivos y módulos:

```plaintext
tech-tutor-engine/
├── .github/
│   └── workflows/
│       └── deploy.yml              # Pipeline CI/CD: Validación sintáctica y sincronización con HF Spaces
├── app/
│   ├── __init__.py
│   ├── main.py                     # Entry point: Lifespan context, configuración CORS y agregación de rutas
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # Configuración tipada global basada en pydantic-settings
│   │   └── security.py             # Middleware de autenticación mediante Bearer Tokens
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py           # Agregador centralizado de endpoints v1
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           ├── health.py       # Sondas de liveness y readiness del sistema (/health)
│   │           ├── query.py        # Validación e introspección directa del RAG (/api/v1/query)
│   │           ├── evaluate.py     # Pipeline multimodal E2E de audio (/api/v1/evaluate/voice)
│   │           └── websocket.py    # Canal full-duplex de streaming de audio (/api/v1/ws/practice)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── asr_service.py          # Wrapper de Faster-Whisper, decodificación CTranslate2 y cálculo WER
│   │   ├── rag_service.py          # Conector a ChromaDB, embeddings all-MiniLM-L6-v2 y filtros semánticos
│   │   └── llm_service.py          # Orquestador LangChain/OpenAI hacia Groq (openai/gpt-oss-120b)
│   └── schemas/
│       ├── __init__.py
│       ├── request_schemas.py      # Esquemas de validación de entrada (QueryRequest, AudioMetadata)
│       └── response_schemas.py     # Esquemas de salida estructurada (EvaluationResponse, HealthResponse)
├── data/
│   └── .gitkeep                    # Directorio local de persistencia vectorial (ignorado en Git VCS)
├── tests/
│   ├── __init__.py
│   ├── test_asr.py                 # Pruebas unitarias sobre flujos de audio en memoria (BytesIO)
│   ├── test_rag.py                 # Validación de distancias coseno y precisión de recuperación semántica
│   ├── test_api.py                 # Pruebas de integración asíncrona sobre FastAPI (pytest-asyncio)
│   └── benchmark_thesis.py         # Script cuantitativo automatizado de latencia, Recall@K y WER
├── .dockerignore                   # Reglas de exclusión para compilación limpia de la imagen Docker
├── .env.example                    # Plantilla de variables de entorno sin secretos expuestos
├── .gitignore                      # Exclusión estricta de artefactos pesados, entornos virtuales y bases vectoriales
├── Dockerfile                      # Empaquetamiento multi-stage optimizado para Linux Debian en HF Spaces
├── README.md                       # Especificación técnica, documentación de endpoints y guía de despliegue
└── requirements.txt                # Manifiesto de dependencias de Python fijadas con precisión

```

---

## 3. Especificación Detallada de los Sub-sistemas

### 3.1. Sub-sistema ASR (Reconocimiento Automático del Habla)

* **Motor Base:** `faster-whisper` ejecutando el modelo `large-v3-turbo` sobre el runtime C++ CTranslate2.


* **Configuración de Hardware y Precisión:** Dispositivo fijado en `device="cpu"` con cuantización entera de 8 bits (`compute_type="int8"`), reduciendo el consumo de memoria del modelo a ~1.6 GB de RAM y garantizando una velocidad de transcripción hasta 4 veces superior respecto al modelo Whisper vanilla en procesadores convencionales.


* **Requisito Crítico del Sistema Operativo:** Dependencia estricta de los binarios del sistema `ffmpeg` para la decodificación de contenedores de audio comprimido (`.webm`, `.mp3`, `.ogg`, `.m4a`). Sin `ffmpeg` en el `PATH` del sistema, la lectura de flujos acústicos genera excepciones irrecuperables.


* **Inyección de Sesgo Léxico (`initial_prompt`):** Para mitigar la tendencia de los modelos acústicos de mapear términos de ingeniería a palabras coloquiales en inglés, se condiciona la atención acústica con la directiva:
```python
INITIAL_PROMPT = (
    "Technical software engineering discussion covering PyTorch, Kubernetes, "
    "Docker, FastAPI, horizontal sharding, microservices, latency, and system design."
)

```


* **Mapeo No Bloqueante en el Servidor Asíncrono:** La inferencia acústica de CTranslate2 es una operación intensiva en CPU. Para evitar congelar el bucle de eventos (`Event Loop`) de FastAPI, la ejecución se delega a un hilo de procesamiento aislado mediante `asyncio.to_thread(asr_service.transcribe, audio_bytes)`.


* **Tratamiento Metodológico del Word Error Rate (WER):**
* *Modo de Habla Libre (`free_speech`):* No se cuenta con una transcripción previa ("Ground Truth") de lo que el ingeniero expresará. La calidad acústica se reporta cuantitativamente mediante la confianza logarítmica promedio (`avg_logprob`), la probabilidad de ausencia de voz (`no_speech_prob`) y el ritmo de elocución medido en palabras por minuto (WPM).


* *Modo Benchmarking / Shadowing (`shadowing_benchmark`):* El payload de la petición recibe de forma opcional el campo `expected_transcript`. Si dicho parámetro se encuentra presente, el servicio ejecuta el algoritmo de Distancia de Levenshtein a nivel de palabras mediante la librería `jiwer` para computar de forma exacta el WER y CER (Character Error Rate).





### 3.2. Sub-sistema RAG y Base de Datos Vectorial

* **Motor Vectorial:** ChromaDB configurado en modo cliente persistente local sobre disco (`PersistentClient`).


* **Colección Activa:** `tech_tutor_corpus`, integrada por un volumen confirmado de **213,439 vectores**.


* **Origen de los Datos ("Ground Truth"):** Documentación técnica oficial, guías de arquitectura y manuales de 31 repositorios de código abierto de GitHub (PyTorch, TensorFlow, AWS CLI, Kubernetes, Docker, FastAPI, System Design Primer, Developer Roadmap, Google Styleguide, etc.).


* **Modelo de Representación Densa (Embeddings):** `all-MiniLM-L6-v2` de Sentence-Transformers, proyectando el texto a un espacio vectorial euclidiano normalizado de 384 dimensiones.


* **Función de Métrica:** Similitud de Coseno (*Cosine Similarity*).


* **Estrategia de Segmentación Original:** Segmentación carácter-recursiva con tamaño de fragmento (`chunk_size`) de 600 caracteres y solapamiento (`chunk_overlap`) de 60 caracteres (10%) para preservar la integridad de definiciones algorítmicas.


* **Sanitización Dinámica Post-Retrieval:** Dado que los fragmentos indexados desde plataformas como Developer Roadmap contienen etiquetas residuales de maquetado (`@article@`, `@video@`, `@feed@`), el servicio `rag_service.py` aplica una limpieza por expresiones regulares sobre los textos recuperados antes de su ensamblaje en el prompt del LLM, ahorrando tokens y depurando la información inyectada.


* **Fusión Semántica con el Escenario:** Para maximizar la relevancia en consultas donde el estudiante emplea expresiones ambiguas, la consulta enviada a ChromaDB se formula como `f"{target_scenario}: {query_text}"`.



### 3.3. Sub-sistema LLM (Razonamiento y Evaluación Técnica)

* **Proveedor de Infraestructura:** Groq Cloud Platform (`[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)`).


* **Identificador del Modelo:** `openai/gpt-oss-120b`.
* **Parámetros de Decodificación:** Temperatura estricta $\tau = 0.1$ para suprimir la dispersión estocástica y penalización de repetición equilibrada.


* **Rol Pedagógico y Operativo:** El modelo asume la identidad de un Evaluador Lingüístico Técnico Senior y Especialista en Arquitectura de Software. Su objetivo es evaluar la intervención del usuario contrastándola contra el contexto técnico recuperado y las directrices del MCER:


1. *Gramática y Sintaxis Técnica:* Precisión en el uso de la voz pasiva, tiempos perfectos y oraciones condicionales empleadas en debates de ingeniería.


2. *Precisión Conceptual y Terminológica:* Sustitución de explicaciones coloquiales por terminología formal de la industria (ej. transformar *"we split the database so it writes fast"* a *"we implement horizontal database sharding to distribute write throughput"* con referencia explícita al repositorio fuente).


3. *Cálculo Determinista de Métricas:* El cálculo final del ratio de densidad léxica técnica no se delega a la aritmética interna del LLM. `gpt-oss-120b` extrae y lista los términos técnicos detectados (`detected_technical_terms`), y una función auxiliar en Python calcula de forma determinista la relación matemática contra el volumen total de tokens de la respuesta del usuario.





---

## 4. Contratos de Datos y Especificación de la API (OpenAPI)

### 4.1. Sonda de Estado y Disponibilidad Operativa

* **Ruta:** `GET /health`

* **Descripción:** Comprueba la disponibilidad en memoria y la inicialización de los tres subsistemas (ASR, ChromaDB y LLM).


* **Autenticación:** `Bearer <API_TOKEN>`

* **Código de Respuesta:** `HTTP 200 OK`


```json
{
  "status": "ok",
  "app": "Tech Tutor Engine",
  "version": "1.0.0",
  "chroma_ready": true,
  "asr_ready": true,
  "llm_configured": true
}

```

### 4.2. Recuperación e Inspección Semántica RAG

* **Ruta:** `POST /api/v1/query`

* **Descripción:** Endpoint directo para pruebas de banco semánticas sin procesamiento de voz.


* **Headers:** `Content-Type: application/json`, `Authorization: Bearer <API_TOKEN>`

* **Request Body:**

```json
{
  "query_text": "How does horizontal database sharding mitigate write bottleneck?",
  "top_k": 3
}

```

* **Response Body (`HTTP 200 OK`):**

```json
{
  "status": "success",
  "query_text": "How does horizontal database sharding mitigate write bottleneck?",
  "top_k": 3,
  "chunks": [
    {
      "document": "Sharding is a type of database partitioning that separates very large databases into smaller, faster, more easily managed parts called data shards...",
      "metadata": {
        "source": "developer-roadmap\\src\\data\\roadmaps\\server-side-game-developer\\content\\sharding.md"
      },
      "distance": 0.3127,
      "source": "developer-roadmap\\src\\data\\roadmaps\\server-side-game-developer\\content\\sharding.md"
    }
  ]
}

```

### 4.3. Pipeline Multimodal de Evaluación de Voz (Core MaaS)

* **Ruta:** `POST /api/v1/evaluate/voice`

* **Headers:** `Content-Type: multipart/form-data`, `Authorization: Bearer <API_TOKEN>`

* **Parámetros del Formulario (`multipart/form-data`):**
* `audio_file`: Archivo binario de voz (`.webm`, `.mp3`, `.wav`).


* `target_scenario`: Identificador del escenario técnico (`system_design_interview`, `code_review`, `incident_postmortem`, `architecture_presentation`).


* `expected_transcript` *(Opcional)*: Texto de referencia técnica en caso de ejecutar una prueba de lectura o benchmarking fonético controlado para el cálculo de WER.




* **Esquema Unificado de Salida (`HTTP 200 OK`):**

```json
{
  "status": "success",
  "execution_time_ms": 1845.2,
  "asr_metrics": {
    "transcription": "We need to shard the database across multiple nodes because we are experiencing high write latency.",
    "detected_language": "en",
    "speech_duration_seconds": 5.4,
    "confidence_score": 0.962,
    "word_count": 15,
    "words_per_minute": 166.7,
    "wer_score": null
  },
  "evaluation_core": {
    "overall_score": 91.5,
    "cefr_level": "C1",
    "lexical_density_ratio": 0.467,
    "grammar_accuracy_score": 94.0,
    "technical_precision_score": 89.0,
    "detected_technical_terms": [
      "shard",
      "database",
      "nodes",
      "write latency"
    ]
  },
  "feedback_details": [
    {
      "type": "vocabulary_enhancement",
      "original_phrase": "shard the database across multiple nodes",
      "suggested_phrase": "implement horizontal sharding across distributed database instances",
      "rag_source_context": "donnemartin/system-design-primer (Sharding Strategies)",
      "explanation": "Using precise system terminology such as 'distributed database instances' reinforces professional seniority during technical evaluations."
    }
  ]
}

```

### 4.4. Canal WebSocket para Streaming de Audio

* **Ruta:** `ws://<dominio>/api/v1/ws/practice`

* **Protocolo de Mensajería:** Transmisión de fragmentos binarios (PCM / WebM) y emisión de eventos de sincronización:
* Evento 1: `{"event": "transcribing"}`

* Evento 2: `{"event": "retrieving", "partial_text": "..."}`

* Evento 3: `{"event": "evaluating"}`

* Evento 4: `{"event": "completed", "payload": { ... EvaluationResponse ... }}`




---

## 5. Implementación de Modelos y Servicios de Código

### 5.1. Modelos de Configuración y Esquemas Pydantic

#### `app/core/config.py`

```python
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Tech Tutor Engine"
    API_V1_STR: str = "/api/v1"
    API_TOKEN: str = "D1mZjnJzKVP_3kKsB8e-vuMYdGzGmxlzTv9nTjVd9zA"
    
    # Proveedor LLM (Groq Cloud)
    GROQ_API_KEY: str
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_MODEL_NAME: str = "openai/gpt-oss-120b"
    LLM_TEMPERATURE: float = 0.1
    
    # Parámetros ASR (Faster-Whisper)
    WHISPER_MODEL_NAME: str = "large-v3-turbo"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    
    # Base Vectorial (ChromaDB)
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/technical_corpus_db")
    CHROMA_COLLECTION_NAME: str = "tech_tutor_corpus"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

```

#### `app/schemas/response_schemas.py`

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class ASRMetrics(BaseModel):
    transcription: str
    detected_language: str
    speech_duration_seconds: float
    confidence_score: float
    word_count: int
    words_per_minute: float
    wer_score: Optional[float] = None

class FeedbackItem(BaseModel):
    type: str = Field(description="Clasificación: vocabulary_enhancement, grammar_correction, architectural_alignment")
    original_phrase: str
    suggested_phrase: str
    rag_source_context: str
    explanation: str

class EvaluationCore(BaseModel):
    overall_score: float = Field(ge=0.0, le=100.0)
    cefr_level: str = Field(description="B1, B2 o C1 de acuerdo con el Marco Común Europeo")
    lexical_density_ratio: float = Field(ge=0.0, le=1.0)
    grammar_accuracy_score: float = Field(ge=0.0, le=100.0)
    technical_precision_score: float = Field(ge=0.0, le=100.0)
    detected_technical_terms: List[str]

class EvaluationResponse(BaseModel):
    status: str = "success"
    execution_time_ms: float
    asr_metrics: ASRMetrics
    evaluation_core: EvaluationCore
    feedback_details: List[FeedbackItem]

```

### 5.2. Servicio de Razonamiento LLM (`app/services/llm_service.py`)

```python
import json
import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
            max_retries=2
        )
        self._setup_prompt_template()

    def _setup_prompt_template(self):
        system_instruction = (
            "You are an elite AI Technical English Evaluator and Senior Software Architect. "
            "Your task is to critically analyze an engineer's spoken technical English response against "
            "authoritative industry documentation and CEFR proficiency rubrics (B2 to C1 levels).\n\n"
            "GROUND TRUTH TECHNICAL CONTEXT:\n{retrieved_context}\n\n"
            "EVALUATION CRITERIA:\n"
            "1. CEFR Level & Grammar: Evaluate syntax complexity, modal verbs, conditionals, and passive voice.\n"
            "2. Technical Precision: Identify colloquial explanations and upgrade them to professional terminology "
            "found in the provided context (e.g., replace 'make it bigger' with 'scale horizontally').\n"
            "3. Identify all valid engineering terms used by the speaker.\n\n"
            "OUTPUT REQUIREMENT:\n"
            "You must return ONLY a strictly valid JSON object matching this schema:\n"
            "{{\n"
            '  "overall_score": <float 0-100>,\n'
            '  "cefr_level": "<B1|B2|C1>",\n'
            '  "grammar_accuracy_score": <float 0-100>,\n'
            '  "technical_precision_score": <float 0-100>,\n'
            '  "detected_technical_terms": ["term1", "term2"],\n'
            '  "feedback_details": [\n'
            "    {{\n"
            '      "type": "vocabulary_enhancement",\n'
            '      "original_phrase": "<speaker phrase>",\n'
            '      "suggested_phrase": "<industry standard phrase>",\n'
            '      "rag_source_context": "<repository source>",\n'
            '      "explanation": "<justification>"\n'
            "    }}\n"
            "  ]\n"
            "}}"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", "Target Scenario: {scenario}\nSpeaker Transcript: {transcript}")
        ])

    async def evaluate_technical_speech(
        self, transcript: str, retrieved_context: str, scenario: str
    ) -> Dict[str, Any]:
        chain = self.prompt | self.llm
        response = await chain.ainvoke({
            "retrieved_context": retrieved_context,
            "scenario": scenario,
            "transcript": transcript
        })
        
        content = response.content.strip()
        # Limpieza de bloques de código markdown si el LLM los incluye
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        return json.loads(content.strip())

llm_service = LLMService()

```

### 5.3. Endpoint Multimodal de Inferencia (`app/api/v1/endpoints/evaluate.py`)

```python
import time
import asyncio
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from app.core.security import verify_api_token
from app.services.asr_service import asr_service
from app.services.rag_service import rag_service
from app.services.llm_service import llm_service
from app.schemas.response_schemas import EvaluationResponse, ASRMetrics, EvaluationCore, FeedbackItem

router = APIRouter()

@router.post(
    "/evaluate/voice",
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Pipeline Multimodal Asíncrono de Evaluación de Voz"
)
async def evaluate_voice(
    audio_file: UploadFile = File(..., description="Archivo de audio binario (.webm, .mp3, .wav)"),
    target_scenario: str = Form(..., description="Escenario técnico (e.g., system_design_interview)"),
    expected_transcript: Optional[str] = Form(None, description="Texto de referencia opcional para calcular WER"),
    token: str = Depends(verify_api_token)
):
    start_time = time.perf_counter()
    
    # 1. Validación de formato de archivo
    valid_extensions = (".webm", ".mp3", ".wav", ".ogg", ".m4a")
    if not audio_file.filename.lower().endswith(valid_extensions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato no soportado. Extensiones válidas: {valid_extensions}"
        )
    
    audio_bytes = await audio_file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Archivo de audio vacío.")

    # 2. Inferencia ASR en hilo secundario (Evita bloquear el Event Loop de FastAPI)
    try:
        asr_data = await asyncio.to_thread(
            asr_service.transcribe_audio,
            audio_bytes=audio_bytes,
            expected_transcript=expected_transcript
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fallo en decodificación acústica ASR: {str(e)}"
        )

    transcript = asr_data["transcription"]
    if not transcript.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No se detectó habla inteligible en la señal de audio proporcionada."
        )

    # 3. Búsqueda Semántica Vectorial en ChromaDB (all-MiniLM-L6-v2)
    search_query = f"{target_scenario}: {transcript}"
    rag_chunks = await asyncio.to_thread(
        rag_service.retrieve_context,
        query_text=search_query,
        top_k=3
    )
    
    formatted_context = "\n---\n".join([
        f"Source: {chunk['metadata'].get('source', 'Unknown')}\nContent: {chunk['document']}"
        for chunk in rag_chunks
    ])

    # 4. Inferencia y Razonamiento Asíncrono con openai/gpt-oss-120b
    try:
        llm_raw_evaluation = await llm_service.evaluate_technical_speech(
            transcript=transcript,
            retrieved_context=formatted_context,
            scenario=target_scenario
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Fallo en la inferencia del motor LLM: {str(e)}"
        )

    # 5. Cálculo Determinista de Densidad Léxica en Backend
    words = [w for w in transcript.split() if len(w) > 1]
    total_words = len(words) if len(words) > 0 else 1
    detected_terms = llm_raw_evaluation.get("detected_technical_terms", [])
    lexical_density = round(min(len(detected_terms) / total_words, 1.0), 3)

    execution_duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return EvaluationResponse(
        status="success",
        execution_time_ms=execution_duration_ms,
        asr_metrics=ASRMetrics(
            transcription=transcript,
            detected_language=asr_data["detected_language"],
            speech_duration_seconds=asr_data["duration_seconds"],
            confidence_score=asr_data["confidence_score"],
            word_count=total_words,
            words_per_minute=round((total_words / (asr_data["duration_seconds"] or 1.0)) * 60, 1),
            wer_score=asr_data.get("wer_score")
        ),
        evaluation_core=EvaluationCore(
            overall_score=float(llm_raw_evaluation.get("overall_score", 0.0)),
            cefr_level=str(llm_raw_evaluation.get("cefr_level", "B1")),
            lexical_density_ratio=lexical_density,
            grammar_accuracy_score=float(llm_raw_evaluation.get("grammar_accuracy_score", 0.0)),
            technical_precision_score=float(llm_raw_evaluation.get("technical_precision_score", 0.0)),
            detected_technical_terms=detected_terms
        ),
        feedback_details=[
            FeedbackItem(**item) for item in llm_raw_evaluation.get("feedback_details", [])
        ]
    )

```

---

## 6. Estrategia de Gestión de Datos y Control de Versiones Git

La persistencia de **213,439 vectores** en `./data/technical_corpus_db` y los binarios de modelos de IA superan con creces los límites de almacenamiento de repositorios regulares en GitHub (máximo 100 MB por archivo individual). Para preservar un repositorio ágil y cumplir las directrices de MLOps, se establece la siguiente separación:

### 6.1. Reglas de Ignorado Estricto (`.gitignore`)

```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Entornos virtuales
.venv/
venv/
env/
ENV/

# Almacén de artefactos y bases de datos locales
data/
*.sqlite3
*.bin
*.pt
*.onnx

# Pesos de modelos pre-descargados
models/
huggingface/
.cache/

# Archivos de entorno y credenciales sensibles
.env
.env.local

# Cobertura de pruebas y logs de ejecución
.coverage
htmlcov/
*.log
.pytest_cache/

```

### 6.2. Estrategia de Ingesta de la Base Vectorial en Producción

Para que el contenedor en Hugging Face Spaces disponga de la colección sin saturar el historial de Git, se define un enfoque dual de aprovisionamiento:

* **Enfoque A (Git LFS en Hugging Face Space):** El Space en Hugging Face se configura con Git LFS habilitado exclusivamente para el archivo consolidado comprimido `technical_corpus_db.tar.gz`. Durante el comando de construcción del contenedor, la base se descomprime en `./data/technical_corpus_db`.


* **Enfoque B (Descarga Automática de Dataset vía Hub API en el Lifespan de FastAPI):** La carpeta vectorial se aloja como un Dataset Privado en Hugging Face (`leonardo-beltran/tech-tutor-corpus`). Durante la fase de inicialización (`lifespan`) en `app/main.py`, el servidor ejecuta la sincronización si el directorio local no existe:


```python
from huggingface_hub import snapshot_download
import os

if not os.path.exists("./data/technical_corpus_db"):
    snapshot_download(
        repo_id="leonardo-beltran/tech-tutor-corpus",
        repo_type="dataset",
        local_dir="./data/technical_corpus_db",
        token=os.getenv("HF_TOKEN")
    )

```



---

## 7. MLOps: Contenerización Docker y Despliegue en Hugging Face Spaces

### 7.1. Manifiesto de Dependencias (`requirements.txt`)

Fijación rigurosa de dependencias para evitar incompatibilidades de enlace dinámico en Linux x86_64:

```text
# Núcleo Web Asíncrono
fastapi>=0.110.0,<1.0.0
uvicorn[standard]>=0.28.0,<1.0.0
pydantic>=2.6.0,<3.0.0
pydantic-settings>=2.2.0,<3.0.0
python-multipart>=0.0.9

# Procesamiento ASR y Modelos Acústicos
setuptools<70.0.0
ctranslate2>=4.1.0
faster-whisper>=1.0.0
jiwer>=3.0.3

# Base de Datos Vectorial y Embeddings
chromadb>=0.4.24
sentence-transformers>=2.5.0
torch--index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

# Orquestación LLM
langchain>=0.1.13
langchain-openai>=0.1.1
httpx>=0.27.0
huggingface-hub>=0.21.4

# Calidad y Pruebas
pytest>=8.0.0
pytest-asyncio>=0.23.0

```

### 7.2. Contenerización Multi-Stage (`Dockerfile`)

El contenedor se diseña en dos fases para minimizar la huella de la imagen en disco, instalar las dependencias del sistema requeridas (`ffmpeg`) y descargar de manera anticipada los pesos de los modelos durante el proceso de empaquetamiento (*build-time*), eliminando el riesgo de que el servidor sufra caídas por límite de tiempo de arranque (*Startup Timeout*):

```dockerfile
# ==============================================================================
# ETAPA 1: BUILDER (Compilación y Resolución de Dependencias)
# ==============================================================================
FROM python:3.11-slim as builder

WORKDIR /install

# Herramientas de compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instalación aislada de dependencias
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==============================================================================
# ETAPA 2: RUNTIME (Entorno Ligero de Ejecución en Hugging Face Spaces)
# ==============================================================================
FROM python:3.11-slim

WORKDIR /app

# Instalación obligatoria de ffmpeg para el pipeline ASR en CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copia de paquetes compilados desde la etapa anterior
COPY --from=builder /install /usr/local

# Configuración de usuario no privilegiado requerido por Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HOME=/home/user/.cache/huggingface

# Pre-descarga de pesos de modelos durante el build (Zero Cold-Start)
RUN python3 -c "\
from faster_whisper import WhisperModel; \
WhisperModel('large-v3-turbo', device='cpu', compute_type='int8', download_root='/home/user/.cache/whisper'); \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/home/user/.cache/huggingface')"

# Copia del código fuente del proyecto
COPY --chown=user:user . /app

# Puerto estándar fijado por Hugging Face Spaces
EXPOSE 7860

# Comando de arranque del servidor Uvicorn sobre el puerto 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

```

### 7.3. Pipeline CI/CD Automatizado (`.github/workflows/deploy.yml`)

Sincronización automatizada desde el repositorio principal de GitHub hacia el Space de Hugging Face:

```yaml
name: Continuous Deployment to Hugging Face Spaces

on:
  push:
    branches: [ "main" ]

jobs:
  test-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          lfs: true

      - name: Set up Python 3.11
        uses: setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install System Dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y ffmpeg

      - name: Install Test Dependencies
        run: |
          pip install -r requirements.txt

      - name: Execute Automated Unit & Integration Tests
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          API_TOKEN: ${{ secrets.API_TOKEN }}
        run: |
          pytest tests/test_api.py tests/test_rag.py -v

      - name: Push to Hugging Face Spaces Remote
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_SPACE_REPO: "[https://huggingface.co/spaces/leonardo-beltran/tech-tutor-engine](https://huggingface.co/spaces/leonardo-beltran/tech-tutor-engine)"
        run: |
          git config --global user.email "leonardo.beltran@uatx.mx"
          git config --global user.name "Leonardo Beltran"
          git remote add space "$HF_SPACE_REPO"
          git push --force https://leonardo-beltran:$HF_TOKEN@huggingface.co/spaces/leonardo-beltran/tech-tutor-engine main:main

```

---

## 8. Protocolo de Pruebas y Validación para la Defensa de Tesis

Para la fundamentación del capítulo de resultados empíricos y análisis de software en el documento escrito de grado, se ejecutará el script de benchmarking automatizado (`tests/benchmark_thesis.py`):

```
+----------------------------------------------------------------------------------------------------+
|                          BANCO DE VALIDACIÓN EXPERIMENTAL (HITO 3 Y 4)                             |
+----------------------------------------------------------------------------------------------------+
| 1. EVALUACIÓN ACÚSTICA (WER / CER)                                                                 |
|    • Ingesta de 50 muestras de audio controladas (inglés técnico con acento hispanohablante).      |
|    • Comparación fonética automatizada contra Ground Truth mediante librería jiwer.                |
|    • Métrica Objetivo: WER < 12% en terminología de arquitectura e infraestructura.                |
|                                                                                                    |
| 2. EVALUACIÓN SEMÁNTICA RAG (Retrieval Recall@K & NDCG)                                            |
|    • Ejecución de 100 consultas técnicas canónicas formuladas por ingenieros de software.          |
|    • Inspección de los Top-3 y Top-5 fragmentos recuperados en tech_tutor_corpus.                 |
|    • Métrica Objetivo: Recall@3 >= 85% sobre documentación oficial de repositorios.                |
|                                                                                                    |
| 3. PERFILAMIENTO DE LATENCIAS E2E                                                                  |
|    • Desglose instrumental de tiempos de respuesta:                                                |
|        - Latencia de Ingesta y Decodificación ASR en CPU (ms).                                     |
|        - Latencia de Búsqueda Vectorial Coseno en ChromaDB (ms).                                   |
|        - Latencia de Inferencia LPU Groq para openai/gpt-oss-120b (ms).                             |
|    • Métrica Objetivo: Tiempo total de procesamiento interactivo < 2.5 segundos.                    |
+----------------------------------------------------------------------------------------------------+

```

### Tabla Comparativa de Rendimiento y Arquitectura (Evidencia para el Sínodo)

| Parámetro / Componente | Configuración Inicial (Prototipo Local)

 | Configuración Final de Producción (MaaS Cloud)

 | Justificación Metodológica de Tesis |
| --- | --- | --- | --- |
| **Entorno de Despliegue** | Localhost (Windows x64 / Port 8000)

 | Hugging Face Spaces (Docker Debian / Port 7860)

 | Reproducibilidad, portabilidad industrial y contratos OpenAPI.

 |
| **Motor de Razonamiento** | Llama 3 (`llama-3.3-70b-versatile`)

 | **`openai/gpt-oss-120b`** (Groq LPU) | Evaluación lingüística y de arquitectura de software superior.

 |
| **Tiempo de Generación LLM** | ~420 ms

 | **~500 ms** (MoE 120B en LPUs) | Capacidad analítica expandida con una mínima variación de latencia.

 |
| **Manejo de Audio (ASR)** | Inferencia Síncrona (Bloqueo de Event Loop)

 | Inferencia Desacoplada (`asyncio.to_thread`)

 | Prevención de caídas de conexión y soporte de concurrencia en FastAPI.

 |
| **Inyección de Dependencias** | Ruedas compiladas locales en Windows

 | Multi-Stage Dockerfile con Pre-caching de Pesos

 | Eliminación de arranques en frío (*Cold-Start Timeouts*) en la nube.

 |
| **Base Vectorial** | 213,439 vectores en almacenamiento local

 | ChromaDB Persistente desacoplado de Git VCS

 | Respeto a los límites de tamaño de repositorios y aislamiento MLOps.

 |

---

## 9. Lista de Verificación Inmediata para Subida a GitHub y Producción

1. **Creación del Repositorio:** Inicializar el repositorio Git local mediante `git init` en el directorio raíz `tech-tutor-engine/`.


2. **Aplicación del `.gitignore`:** Comprobar mediante `git status` que la carpeta `./data/technical_corpus_db`, los entornos virtuales `.venv` y los archivos `.env` permanezcan ignorados de forma estricta.


3. **Commit Inicial y Push:**
```bash
git add app/ tests/ .github/ Dockerfile requirements.txt .gitignore .env.example README.md
git commit -m "feat(core): initial production release for MaaS Tech Tutor Engine with gpt-oss-120b"
git branch -M main
git remote add origin [https://github.com/leo7leoavil-ship-it/tech_tutor_engine_V2.git](https://github.com/leo7leoavil-ship-it/tech_tutor_engine_V2.git)
git push -u origin main

```


4. **Configuración de Secretos en Hugging Face Space:**
* `GROQ_API_KEY`: Clave de acceso a Groq Cloud con permisos para invocar `openai/gpt-oss-120b`.
* `API_TOKEN`: Token Bearer establecido para proteger los endpoints de la API (`D1mZjnJzKVP_3kKsB8e-vuMYdGzGmxlzTv9nTjVd9zA`).




5. **Verificación de Despliegue en Vivo:**
* Acceder a `https://<tu-space>.hf.space/docs` para validar la interfaz Swagger generada automáticamente.


* Ejecutar la prueba de salud emitiendo `curl -X GET https://<tu-space>.hf.space/health -H "Authorization: Bearer <API_TOKEN>"`.