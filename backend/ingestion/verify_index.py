"""
verify_index.py
---------------
Sanity-checks the FAISS and BM25 indexes after indexer.py has run.

Run:
    python backend/ingestion/verify_index.py
    python backend/ingestion/verify_index.py --queries
    python backend/ingestion/verify_index.py --query "your query here"
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.embeddings import get_embedding_service


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else BACKEND_ROOT / path


SETTINGS_FILE = BACKEND_ROOT / "settings.yaml"
DOCUMENTS_FILE = BACKEND_ROOT / "extracted" / "documents.json"
ACRONYMS_FILE = BACKEND_ROOT / "extracted" / "acronyms.json"
PARENT_INDEX_FILE = BACKEND_ROOT / "extracted" / "parent_index.json"
BM25_FILE = BACKEND_ROOT / "db" / "bm25_index.pkl"
CAPTIONS_CACHE = BACKEND_ROOT / "extracted" / "image_captions.json"

with open(SETTINGS_FILE, encoding="utf-8") as handle:
    CFG = yaml.safe_load(handle)

RETRIEVAL_CFG = CFG["retrieval"]
OLLAMA_CFG = CFG["ollama"]
MODEL_CFG = CFG["models"]

FAISS_INDEX = resolve_repo_path(RETRIEVAL_CFG["faiss_index_path"])
FAISS_ID_MAP = resolve_repo_path(RETRIEVAL_CFG["faiss_id_map_path"])
HNSW_M = RETRIEVAL_CFG["hnsw_M"]
HNSW_EF_CONSTRUCTION = RETRIEVAL_CFG["hnsw_ef_construction"]
HNSW_EF_SEARCH = RETRIEVAL_CFG["hnsw_ef_search"]
OLLAMA_BASE = OLLAMA_CFG.get("base_url", "http://localhost:11434")
EMBED_MODEL = MODEL_CFG["embedding"]
EMBEDDING_SERVICE = get_embedding_service()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SAMPLE_QUERIES = [
    ("factual", "What is the definition of a system in NASA systems engineering?"),
    ("process", "What are the key reviews in the NASA project life cycle phases?"),
    ("acronym", "What does CDR stand for?"),
    ("parent", "Summarize section 3.0 NASA Program/Project Life Cycle"),
    ("figure", "Describe the V-model diagram used in systems engineering verification"),
]

def embed(text: str) -> np.ndarray:
    return EMBEDDING_SERVICE.embed_query(text)


def normalize_id_map(raw: Any) -> dict[str, dict[str, Any]]:
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
    if isinstance(raw, list):
        return {str(i): v for i, v in enumerate(raw) if isinstance(v, dict)}
    return {}


def entry_text(entry: dict[str, Any]) -> str:
    return str(entry.get("text_for_embedding") or entry.get("content") or entry.get("text") or "")


def load_index() -> faiss.Index:
    index = faiss.read_index(str(FAISS_INDEX))
    index.hnsw.efSearch = HNSW_EF_SEARCH
    return index


def load_id_map() -> dict[str, dict[str, Any]]:
    with open(FAISS_ID_MAP, encoding="utf-8") as handle:
        return normalize_id_map(json.load(handle))


def expected_corpus_count() -> tuple[int, dict[str, int]]:
    counts = {"document": 0, "acronym": 0, "parent": 0, "image_caption": 0}

    if DOCUMENTS_FILE.exists():
        with open(DOCUMENTS_FILE, encoding="utf-8") as f:
            counts["document"] = len(json.load(f))

    if ACRONYMS_FILE.exists():
        with open(ACRONYMS_FILE, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                counts["acronym"] = len([k for k, v in data.items() if str(v).strip()])

    if PARENT_INDEX_FILE.exists():
        with open(PARENT_INDEX_FILE, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                counts["parent"] = len(data)

    if CAPTIONS_CACHE.exists():
        with open(CAPTIONS_CACHE, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                counts["image_caption"] = len([k for k, v in data.items() if str(v).strip()])

    return counts["document"] + counts["acronym"] + counts["parent"] + counts["image_caption"], counts


def check_faiss() -> bool:
    if not FAISS_INDEX.exists():
        log.error("FAIL  FAISS index not found: %s", FAISS_INDEX)
        return False

    if not FAISS_ID_MAP.exists():
        log.error("FAIL  FAISS id map not found: %s", FAISS_ID_MAP)
        return False

    index = load_index()
    id_map = load_id_map()
    count = int(index.ntotal)

    log.info("PASS  FAISS index exists - %d vectors", count)

    expected_total, expected_by_type = expected_corpus_count()
    if expected_total:
        if count == expected_total:
            log.info("PASS  Vector count matches expected unified corpus count (%d)", expected_total)
        else:
            log.warning("WARN  Vector count %d != expected unified corpus count %d", count, expected_total)

    if len(id_map) == count:
        log.info("PASS  FAISS id map count matches index vectors (%d)", count)
    else:
        log.warning("WARN  id map count %d != FAISS vectors %d", len(id_map), count)

    if getattr(index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT:
        log.info("PASS  FAISS metric is inner product for cosine search")
    else:
        log.warning("WARN  FAISS metric is %s (expected %s)", getattr(index, "metric_type", None), faiss.METRIC_INNER_PRODUCT)

    if int(getattr(index, "d", -1)) == EMBEDDING_SERVICE.dimension:
        log.info("PASS  FAISS dimension matches embedding config (%d)", EMBEDDING_SERVICE.dimension)
    else:
        log.warning(
            "WARN  FAISS dimension is %s (expected %d)",
            getattr(index, "d", None),
            EMBEDDING_SERVICE.dimension,
        )

    class_name = type(index).__name__
    if "HNSW" in class_name.upper():
        log.info("PASS  FAISS index type is %s", class_name)
    else:
        log.warning("WARN  FAISS index type is %s (expected HNSW)", class_name)

    if getattr(index.hnsw, "efConstruction", None) == HNSW_EF_CONSTRUCTION:
        log.info("PASS  hnsw.efConstruction = %s", HNSW_EF_CONSTRUCTION)
    else:
        log.warning("WARN  hnsw.efConstruction = %s (expected %s)", getattr(index.hnsw, "efConstruction", None), HNSW_EF_CONSTRUCTION)

    if getattr(index.hnsw, "efSearch", None) == HNSW_EF_SEARCH:
        log.info("PASS  hnsw.efSearch = %s", HNSW_EF_SEARCH)
    else:
        log.warning("WARN  hnsw.efSearch = %s (expected %s)", getattr(index.hnsw, "efSearch", None), HNSW_EF_SEARCH)

    log.info("INFO  Configured hnsw.M = %s", HNSW_M)

    found: dict[str, int] = {}
    for entry in id_map.values():
        t = str(entry.get("type") or "document")
        found[t] = found.get(t, 0) + 1

    log.info("INFO  Indexed type distribution: %s", found)
    log.info("INFO  Expected type distribution: %s", expected_by_type)

    for t in ("document", "acronym", "parent", "image_caption"):
        if expected_by_type.get(t, 0) > 0 and found.get(t, 0) == 0:
            log.warning("WARN  Expected %s entries but found none in FAISS id map", t)

    return True


def check_bm25() -> bool:
    if not BM25_FILE.exists():
        log.error("FAIL  BM25 index not found: %s", BM25_FILE)
        return False

    try:
        with open(BM25_FILE, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        log.error("FAIL  Could not load BM25 pkl: %s", exc)
        return False

    chunks = payload.get("chunks", [])
    bm25 = payload.get("bm25")
    if not isinstance(bm25, BM25Okapi):
        log.warning("WARN  BM25 payload does not contain BM25Okapi instance")

    log.info("PASS  BM25 index loaded - %d corpus entries", len(chunks))

    if chunks:
        query_tokens = "nasa project life cycle phases".split()
        scores = bm25.get_scores(query_tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
        log.info("INFO  BM25 top-3 for 'nasa project life cycle phases':")
        for idx in top_idx:
            entry = chunks[idx] if isinstance(chunks[idx], dict) else {}
            metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
            etype = str(entry.get("type") or "document")
            title = metadata.get("title", "?")
            log.info("      [score=%.4f] type=%s title=%s", scores[idx], etype, title)

    return True


def check_captions() -> None:
    if not CAPTIONS_CACHE.exists():
        log.info("INFO  No caption cache found (index still valid without captions)")
        return

    with open(CAPTIONS_CACHE, encoding="utf-8") as f:
        captions = json.load(f)

    log.info("INFO  Caption cache: %d image(s) captioned", len(captions))


def run_query(query: str, intent: str = "manual") -> None:
    log.info("-" * 60)
    log.info("QUERY [%s]: %s", intent, query)

    index = load_index()
    id_map = load_id_map()
    q_emb = embed(query).reshape(1, -1)
    scores, indices = index.search(q_emb, 5)

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx == -1:
            continue
        entry = id_map.get(str(idx), {})
        metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
        etype = str(entry.get("type") or "document")
        title = metadata.get("title", "?")
        page = metadata.get("page_start", 0)
        preview = entry_text(entry)[:120].replace("\n", " ")
        log.info("  #%d score=%.4f type=%-8s title=%s page=%s", rank, score, etype, title, page)
        log.info("     %s", preview)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify handbook indexes")
    p.add_argument("--queries", action="store_true", help="Run all sample queries")
    p.add_argument("--query", type=str, help="Run a custom query")
    p.add_argument("--no-faiss-check", action="store_true")
    p.add_argument("--no-bm25-check", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log.info("=== Index Verification ===")

    all_ok = True

    if not args.no_faiss_check:
        all_ok &= check_faiss()

    if not args.no_bm25_check:
        all_ok &= check_bm25()

    check_captions()

    if args.queries:
        log.info("Running %d sample queries ...", len(SAMPLE_QUERIES))
        for intent, query in SAMPLE_QUERIES:
            run_query(query, intent)

    if args.query:
        run_query(args.query, intent="custom")

    log.info("-" * 60)
    if all_ok:
        log.info("RESULT  All checks passed")
    else:
        log.error("RESULT  Some checks failed - review warnings above")
        sys.exit(1)


if __name__ == "__main__":
    main()
