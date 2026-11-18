"""RAG (Retrieval-Augmented Generation) Service.

Manages persistent ChromaDB vector storage and SentenceTransformer embedding
generation for grounding LLM technical evaluations in canonical engineering literature.
"""

import os
import re
import asyncio
from typing import Any, Dict, List, Optional
import numpy as np
from app.core.config import get_settings
from app.core.logging import logger

# Layout tags left over by sources such as roadmap.sh (@article@, @video@, @feed@, ...).
_LAYOUT_TAG_RE = re.compile(r"@[A-Za-z0-9_.\-]+@")
_SPACES_RE = re.compile(r"[ \t]+")
_LINE_EDGE_SPACES_RE = re.compile(r" ?\n ?")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def sanitize_chunk_text(text: str) -> str:
    """Remove residual markup tags from a retrieved chunk to save LLM tokens."""
    if not text:
        return ""
    cleaned = _LAYOUT_TAG_RE.sub("", text)
    cleaned = _SPACES_RE.sub(" ", cleaned)
    cleaned = _LINE_EDGE_SPACES_RE.sub("\n", cleaned)
    cleaned = _BLANK_LINES_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def build_search_query(text: str, scenario: Optional[str] = None) -> str:
    """Fuse scenario and text as ``"{scenario}: {text}"`` (spec 3.2) when a scenario is given."""
    text = text.strip()
    return f"{scenario}: {text}" if scenario else text


class RAGService:
    """Singleton service for vector similarity search over the technical corpus."""

    _instance: Optional["RAGService"] = None

    def __new__(cls) -> "RAGService":
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.settings = get_settings()
        self.persist_directory = os.path.abspath(self.settings.CHROMA_PERSIST_DIRECTORY)
        self.collection_name = self.settings.CHROMA_COLLECTION_NAME
        self.model_name = self.settings.EMBEDDING_MODEL_NAME

        self.client: Optional[Any] = None
        self.collection: Optional[Any] = None
        self.embedding_model: Optional[Any] = None

        self._initialize_resources()
        self._initialized = True

    def _initialize_resources(self) -> None:
        """Initialize ChromaDB PersistentClient and SentenceTransformer."""
        os.makedirs(self.persist_directory, exist_ok=True)
        logger.info(
            f"Initializing RAG Service at '{self.persist_directory}', collection='{self.collection_name}'"
        )

        # 1. Initialize SentenceTransformer (lazy import keeps test startup instant)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            SentenceTransformer = None  # type: ignore

        if SentenceTransformer is not None:
            try:
                logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                self.embedding_model = SentenceTransformer(self.model_name)
                logger.info(f"SentenceTransformer '{self.model_name}' loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer '{self.model_name}': {e}")
                self.embedding_model = None
        else:
            logger.warning("sentence-transformers package not installed yet.")

        # 2. Initialize ChromaDB
        try:
            import chromadb  # type: ignore
        except ImportError:
            chromadb = None  # type: ignore

        if chromadb is not None:
            try:
                self.client = chromadb.PersistentClient(path=self.persist_directory)
                # Ensure collection exists
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                doc_count = self.collection.count()
                logger.info(
                    f"ChromaDB connected successfully. Collection '{self.collection_name}' contains {doc_count} documents."
                )

                # If collection is empty, seed initial baseline technical corpus
                if doc_count == 0:
                    self._seed_baseline_corpus()
            except Exception as e:
                logger.error(f"Error initializing ChromaDB persistent client: {e}")
                self.collection = None
        else:
            logger.warning("chromadb package not installed yet.")

    def _seed_baseline_corpus(self) -> None:
        """Seed foundational software engineering corpus if database is completely empty."""
        logger.info("Seeding foundational baseline engineering snippets into ChromaDB...")
        sample_docs = [
            (
                "doc_sys_001",
                "Distributed System Design: In-memory caching layers such as Redis or Memcached mitigate latency "
                "by avoiding redundant disk I/O. Idempotency guarantees that multiple identical API requests produce "
                "the exact same side effect, which is critical for fault-tolerant microservice architectures.",
                {"source": "Distributed Systems Patterns & RFC 7231", "scenario": "System Design"},
            ),
            (
                "doc_code_002",
                "Code Review and Refactoring: Cyclomatic complexity should be minimized by decomposing monolithic functions "
                "into single-responsibility units. Utilizing immutable data structures and explicit error propagation "
                "prevents side effects and concurrency race conditions.",
                {"source": "Clean Architecture & Effective Engineering Practices", "scenario": "Code Review"},
            ),
            (
                "doc_thesis_003",
                "Academic Defense and Methodology: Empirical evaluation requires rigorous benchmarking against baseline "
                "metrics, statistical significance tests, and detailed profiling of CPU/GPU throughput and memory footprint "
                "under simulated high-concurrency workloads.",
                {"source": "IEEE Computer Society Academic Research Guidelines", "scenario": "Thesis Defense"},
            ),
            (
                "doc_db_004",
                "Database Architecture: Database schema normalization up to Boyce-Codd Normal Form (BCNF) eliminates "
                "redundancy, while strategic denormalization or read replicas optimize read-heavy analytical query throughput.",
                {"source": "Database Internals & Relational Design", "scenario": "System Design"},
            ),
            (
                "doc_async_005",
                "Asynchronous Event Processing: Decoupling components using message brokers like Apache Kafka or RabbitMQ "
                "enables backpressure management, horizontal scaling, and eventual consistency across distributed domains.",
                {"source": "Enterprise Integration Patterns", "scenario": "System Design"},
            ),
        ]

        try:
            ids = [item[0] for item in sample_docs]
            documents = [item[1] for item in sample_docs]
            metadatas = [item[2] for item in sample_docs]

            if self.embedding_model is not None:
                embeddings = self.embedding_model.encode(documents).tolist()
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
            else:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
            logger.info(f"Successfully seeded {len(sample_docs)} baseline technical corpus documents.")
        except Exception as e:
            logger.warning(f"Could not seed baseline corpus: {e}")

    def query_corpus(
        self, text: str, top_k: int = 3, scenario: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query ChromaDB collection using cosine similarity on generated embeddings.
        
        Args:
            text: Query string or candidate response to match against technical corpus.
            top_k: Number of nearest neighbor documents to retrieve (default: 3).
            scenario: Optional scenario identifier fused into the query as "scenario: text".

        Returns:
            List of retrieved snippets with sanitized document text, cosine distance/similarity
            and source metadata.
        """
        if not text or not text.strip():
            return []

        results: List[Dict[str, Any]] = []

        if self.collection is None or self.embedding_model is None:
            logger.warning("RAG Service not fully initialized. Using fallback corpus context.")
            return self._fallback_context(text, top_k)

        try:
            query_embedding = self.embedding_model.encode(
                [build_search_query(text, scenario)]
            ).tolist()
            doc_count = self.collection.count()
            if doc_count == 0:
                return self._fallback_context(text, top_k)

            n_results = min(top_k, doc_count)
            query_response = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            docs = query_response.get("documents", [[]])[0]
            metas = query_response.get("metadatas", [[]])[0]
            distances = query_response.get("distances", [[]])[0]

            for i in range(len(docs)):
                doc_text = sanitize_chunk_text(docs[i])
                metadata = metas[i] if i < len(metas) and metas[i] else {}
                # Chroma cosine distance lies in [0, 2]; similarity is reported clipped to [0, 1].
                raw_dist = float(distances[i]) if i < len(distances) else 0.5
                similarity = max(0.0, min(1.0, 1.0 - raw_dist))
                source = metadata.get("source", "Technical Engineering Corpus")

                results.append(
                    {
                        "document": doc_text,
                        "similarity": round(similarity, 4),
                        "distance": round(raw_dist, 4),
                        "metadata": metadata,
                        "source": source,
                    }
                )

            return results
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}", exc_info=True)
            return self._fallback_context(text, top_k)

    async def aquery_corpus(
        self, text: str, top_k: int = 3, scenario: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Asynchronous non-blocking wrapper around query_corpus."""
        return await asyncio.to_thread(self.query_corpus, text, top_k, scenario)

    def _fallback_context(self, text: str, top_k: int) -> List[Dict[str, Any]]:
        """Graceful fallback documents when ChromaDB collection is empty or offline."""
        defaults = [
            {
                "document": "In distributed architectures, caching layers mitigate database latency and reduce unnecessary I/O. Asynchronous decoupling ensures eventual consistency and fault tolerance.",
                "similarity": 0.88,
                "distance": 0.12,
                "metadata": {"source": "System Design Primer & RFC 7231"},
                "source": "System Design Primer & RFC 7231",
            },
            {
                "document": "Idiomatic technical English in code reviews emphasizes single-responsibility principles, immutability, deterministic error handling, and thread safety.",
                "similarity": 0.85,
                "distance": 0.15,
                "metadata": {"source": "Clean Architecture Standard"},
                "source": "Clean Architecture Standard",
            },
            {
                "document": "Rigorous thesis defenses require precision when discussing empirical methodology, algorithmic complexity (Big O notation), throughput, and benchmark criteria.",
                "similarity": 0.82,
                "distance": 0.18,
                "metadata": {"source": "ACM/IEEE Research Guidelines"},
                "source": "ACM/IEEE Research Guidelines",
            },
        ]
        return defaults[:top_k]

    def is_ready(self) -> bool:
        """Check if RAG service components are operational."""
        return self.collection is not None and self.embedding_model is not None


_rag_singleton: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Retrieve singleton instance of RAGService."""
    global _rag_singleton
    if _rag_singleton is None:
        _rag_singleton = RAGService()
    return _rag_singleton
