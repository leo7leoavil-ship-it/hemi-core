# ==============================================================================
# ETAPA 1: BUILDER (Compilación y resolución de dependencias)
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /install

# Herramientas de compilación (sólo en esta etapa; no llegan a la imagen final)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Ruedas de PyTorch para CPU (evita descargar ~2 GB de librerías CUDA)
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .

# Instalación aislada de dependencias
RUN pip install --prefix=/install -r requirements.txt

# ==============================================================================
# ETAPA 2: RUNTIME (Entorno ligero de ejecución en Hugging Face Spaces)
# ==============================================================================
FROM python:3.11-slim

# ffmpeg es obligatorio para decodificar .webm/.mp3/.ogg/.m4a en el pipeline ASR
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Paquetes compilados desde la etapa anterior
COPY --from=builder /install /usr/local

# Usuario no privilegiado (uid 1000) requerido por Hugging Face Spaces
RUN useradd -m -u 1000 user \
    && mkdir -p /app/data/technical_corpus_db \
    && chown -R user:user /app

USER user
WORKDIR /app

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HOME=/home/user/.cache/huggingface \
    HOST=0.0.0.0 \
    PORT=7860 \
    ENVIRONMENT=production

# Pre-descarga de pesos durante el build (Zero Cold-Start). Ambos modelos se guardan en
# HF_HOME, la MISMA caché que usa la aplicación en tiempo de ejecución.
RUN python3 -c "\
from faster_whisper.utils import download_model; \
download_model('large-v3-turbo'); \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

# Código fuente (los secretos llegan como variables de entorno del Space, nunca en la imagen)
COPY --chown=user:user . /app

# Puerto estándar fijado por Hugging Face Spaces
EXPOSE 7860

# Un único worker: los modelos residen en memoria y se comparten entre peticiones
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
