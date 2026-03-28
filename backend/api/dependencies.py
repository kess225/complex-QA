from __future__ import annotations

from functools import lru_cache

from backend.retriever.rag_chain import RAGChain


@lru_cache(maxsize=1)
def get_rag_chain() -> RAGChain:
    return RAGChain()
