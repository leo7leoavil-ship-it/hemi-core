"""Vector-database provisioning (spec section 6.2, "Enfoque B").

The persisted ChromaDB directory (213,439 vectors) is far too large for Git, so it is
hosted as a (private) dataset on the Hugging Face Hub and pulled on demand during the
FastAPI lifespan, *before* the RAG service opens the ``PersistentClient``.
"""

import os
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.logging import logger

CHROMA_SQLITE_FILE = "chroma.sqlite3"


def corpus_present(persist_dir: str) -> bool:
    """True when the persisted ChromaDB directory already holds a database file."""
    return os.path.isfile(os.path.join(persist_dir, CHROMA_SQLITE_FILE))


def ensure_corpus_available(settings: Optional[Settings] = None) -> bool:
    """Download the vector DB from the HF Hub if it is not on disk.

    Returns:
        True if a database is available after the call, False otherwise (the RAG service
        will then fall back to its tiny baseline corpus and log a warning).
    """
    settings = settings or get_settings()
    persist_dir = os.path.abspath(settings.CHROMA_PERSIST_DIRECTORY)

    if corpus_present(persist_dir):
        logger.info(f"Vector DB found at '{persist_dir}'. Skipping corpus download.")
        return True

    if not settings.CORPUS_AUTO_DOWNLOAD or not settings.CORPUS_DATASET_REPO:
        logger.warning(
            f"Vector DB missing at '{persist_dir}' and automatic download is disabled."
        )
        return False

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub is not installed; cannot download the corpus.")
        return False

    logger.info(
        f"Vector DB missing. Downloading dataset '{settings.CORPUS_DATASET_REPO}' "
        f"into '{persist_dir}' (this can take several minutes)..."
    )
    try:
        os.makedirs(persist_dir, exist_ok=True)
        snapshot_download(
            repo_id=settings.CORPUS_DATASET_REPO,
            repo_type="dataset",
            local_dir=persist_dir,
            token=settings.HF_TOKEN or None,
        )
    except Exception as e:
        logger.error(f"Corpus download failed: {e}", exc_info=True)
        return False

    ready = corpus_present(persist_dir)
    if ready:
        logger.info("Vector DB downloaded successfully.")
    else:
        logger.error(
            f"Dataset downloaded but '{CHROMA_SQLITE_FILE}' was not found in '{persist_dir}'. "
            "Upload the CONTENTS of technical_corpus_db/ at the dataset root."
        )
    return ready
