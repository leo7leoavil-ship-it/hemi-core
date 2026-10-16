"""Corpus Ingestion & Seeding Utility for ChromaDB.

Designed to ingest and index large-scale technical corpora (up to 213,439+ chunks)
into ChromaDB using SentenceTransformer (all-MiniLM-L6-v2) with batched embeddings.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import get_settings
from app.core.logging import logger
from app.services.rag_service import get_rag_service


def batch_ingest(
    chunks: List[Dict[str, Any]],
    batch_size: int = 500,
) -> None:
    """Ingest a list of document dicts into ChromaDB in batches.
    
    Each chunk dict should contain:
    - 'id': str
    - 'text': str
    - 'metadata': dict (e.g. {'source': '...', 'scenario': '...'})
    """
    rag = get_rag_service()
    if rag.collection is None or rag.embedding_model is None:
        logger.error("RAG Service is not initialized. Cannot ingest corpus.")
        return

    total = len(chunks)
    logger.info(f"Starting batch ingestion of {total} chunks (batch_size={batch_size})...")

    start_time = time.perf_counter()

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        ids = [item["id"] for item in batch]
        documents = [item["text"] for item in batch]
        metadatas = [item.get("metadata", {}) for item in batch]

        logger.info(f"Encoding batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}...")
        embeddings = rag.embedding_model.encode(documents, show_progress_bar=False).tolist()

        rag.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info(f"Indexed {min(i + batch_size, total)}/{total} fragments.")

    elapsed = time.perf_counter() - start_time
    logger.info(
        f"Successfully ingested {total} fragments in {elapsed:.2f}s "
        f"({total / max(0.001, elapsed):.1f} docs/sec)."
    )


if __name__ == "__main__":
    logger.info("Verifying ChromaDB collection state...")
    rag_service = get_rag_service()
    if rag_service.collection:
        count = rag_service.collection.count()
        logger.info(f"Current collection '{rag_service.collection_name}' count: {count} documents.")
    else:
        logger.warning("Collection not reachable.")
