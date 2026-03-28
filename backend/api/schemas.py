from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query text")
    memory_str: str | None = Field(default=None, description="Optional conversation memory")


class SourceMetadata(BaseModel):
    chunk_id: str
    entry_type: str
    source: str
    page: int
    page_end: int
    parent_id: str | None = None
    parent_title: str | None = None
    rrf_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


class SourceDocument(BaseModel):
    page_content: str
    metadata: SourceMetadata


class QueryResponse(BaseModel):
    answer: str
    documents: list[SourceDocument]
