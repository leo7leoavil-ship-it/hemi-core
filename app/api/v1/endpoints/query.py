"""Direct RAG introspection endpoint (no voice, no LLM).

Used to validate semantic retrieval quality (Recall@K) against the technical corpus.
"""

from fastapi import APIRouter, Depends, status
from app.core.security import verify_api_token
from app.schemas.request_schemas import QueryRequest
from app.schemas.response_schemas import QueryResponse, RetrievedChunk
from app.services.rag_service import RAGService, get_rag_service

router = APIRouter(tags=["Semantic Retrieval"])


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic search over the technical corpus (ChromaDB)",
)
async def query_corpus_endpoint(
    request: QueryRequest,
    token: str = Depends(verify_api_token),
    rag: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    """Return the top-k most similar (sanitized) chunks with cosine distance and source."""
    hits = await rag.aquery_corpus(request.query_text, top_k=request.top_k)
    return QueryResponse(
        status="success",
        query_text=request.query_text,
        top_k=request.top_k,
        chunks=[
            RetrievedChunk(
                document=hit["document"],
                metadata=hit.get("metadata", {}) or {},
                distance=float(hit.get("distance", 1.0 - float(hit.get("similarity", 0.0)))),
                source=hit.get("source", "Technical Engineering Corpus"),
            )
            for hit in hits
        ],
    )
