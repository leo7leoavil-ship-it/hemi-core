"""Unit tests for the RAG service helpers and retrieval logic (fake Chroma + embedder)."""

import numpy as np

from app.services.rag_service import RAGService, build_search_query, sanitize_chunk_text


class FakeEmbedder:
    def __init__(self):
        self.encoded = []

    def encode(self, texts):
        self.encoded.extend(texts)
        return np.zeros((len(texts), 384))


class FakeCollection:
    def count(self):
        return 10

    def query(self, query_embeddings, n_results, include):
        return {
            "documents": [["@article@Sharding splits data.\n\n\n\n@video@  Watch this", "Second chunk"]],
            "metadatas": [[{"source": "roadmap/sharding.md"}, None]],
            "distances": [[0.3127, 0.5]],
        }


def make_rag() -> RAGService:
    rag = object.__new__(RAGService)
    rag.collection = FakeCollection()
    rag.embedding_model = FakeEmbedder()
    return rag


def test_sanitize_removes_layout_tags_and_collapses_whitespace():
    raw = "@article@Sharding splits data.\n\n\n\n@video@  Watch   this @feed@"
    assert sanitize_chunk_text(raw) == "Sharding splits data.\n\nWatch this"


def test_build_search_query_fusion():
    assert build_search_query(" sharding ", "system_design_interview") == "system_design_interview: sharding"
    assert build_search_query(" sharding ") == "sharding"


def test_query_corpus_returns_distance_sanitized_docs_and_fused_query():
    rag = make_rag()
    hits = rag.query_corpus("write bottleneck", top_k=2, scenario="code_review")

    assert rag.embedding_model.encoded == ["code_review: write bottleneck"]
    assert hits[0]["distance"] == 0.3127
    assert hits[0]["similarity"] == 0.6873
    assert "@article@" not in hits[0]["document"] and "@video@" not in hits[0]["document"]
    assert hits[0]["source"] == "roadmap/sharding.md"
    assert hits[1]["source"] == "Technical Engineering Corpus"  # metadata None -> default


def test_empty_query_returns_nothing():
    assert make_rag().query_corpus("   ") == []


def test_cosine_distances_are_ordered_for_recall_checks():
    hits = make_rag().query_corpus("anything", top_k=2)
    assert [h["distance"] for h in hits] == sorted(h["distance"] for h in hits)
