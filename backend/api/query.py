from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.dependencies import get_rag_chain
from backend.api.schemas import QueryRequest, QueryResponse, SourceDocument, SourceMetadata

router = APIRouter(tags=["query"])


def _to_source_document(doc) -> SourceDocument:
    metadata = doc.metadata or {}
    return SourceDocument(
        page_content=doc.page_content,
        metadata=SourceMetadata(
            chunk_id=str(metadata.get("chunk_id", "")),
            entry_type=str(metadata.get("entry_type", "document")),
            source=str(metadata.get("source", "unknown")),
            page=int(metadata.get("page", 0) or 0),
            page_end=int(metadata.get("page_end", 0) or 0),
            parent_id=(str(metadata["parent_id"]) if metadata.get("parent_id") is not None else None),
            parent_title=(
                str(metadata["parent_title"]) if metadata.get("parent_title") is not None else None
            ),
            rrf_score=float(metadata.get("rrf_score", 0.0) or 0.0),
            dense_rank=(int(metadata["dense_rank"]) if metadata.get("dense_rank") is not None else None),
            sparse_rank=(int(metadata["sparse_rank"]) if metadata.get("sparse_rank") is not None else None),
        ),
    )


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    query_text = payload.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        chain = get_rag_chain()
        prepared = chain.prepare(query_text, memory_str=payload.memory_str)
        answer = chain.invoke(query_text, memory_str=payload.memory_str)
        docs = prepared.get("documents") or []
        source_docs = [_to_source_document(doc) for doc in docs]
        return QueryResponse(answer=answer, documents=source_docs)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
