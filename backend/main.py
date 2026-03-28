from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.dependencies import get_rag_chain
from backend.api.query import router as query_router

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm up the chain once at startup so first query has lower latency.
    try:
        get_rag_chain()
    except Exception as exc:
        # Keep API booting even if model/index dependencies are temporarily unavailable.
        log.warning("RAG chain warmup failed at startup: %s", exc)
    yield


app = FastAPI(
    title="NASA RAG API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(query_router, prefix="/api")
