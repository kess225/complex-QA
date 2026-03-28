"""
indexer.py
----------
Builds a persistent FAISS index plus BM25 corpus from:
  - backend/extracted/documents.json
  - backend/extracted/acronyms.json
  - backend/extracted/parent_index.json
    - backend/extracted/image_captions.json

Run:
    python backend/ingestion/indexer.py
    python backend/ingestion/indexer.py --skip-image-captions
    python backend/ingestion/indexer.py --reset
"""

import argparse
import json
import logging
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Paths / config
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
CAPTIONS_CACHE = BACKEND_ROOT / "extracted" / "image_captions.json"
BM25_FILE = BACKEND_ROOT / "db" / "bm25_index.pkl"
BATCH_SIZE = 50
IMAGE_NAME_RE = re.compile(r"\[Image:\s*([^\]]+)\]")

with open(SETTINGS_FILE, encoding="utf-8") as handle:
    CFG = yaml.safe_load(handle)

RETRIEVAL_CFG = CFG["retrieval"]
OLLAMA_CFG = CFG["ollama"]
MODEL_CFG = CFG["models"]

FAISS_INDEX = resolve_repo_path(RETRIEVAL_CFG["faiss_index_path"])
FAISS_ID_MAP = resolve_repo_path(RETRIEVAL_CFG["faiss_id_map_path"])
FAISS_CORPUS_JSON = FAISS_ID_MAP.with_name("corpus.json")
HNSW_M = RETRIEVAL_CFG["hnsw_M"]
HNSW_EF_CONSTRUCTION = RETRIEVAL_CFG["hnsw_ef_construction"]
HNSW_EF_SEARCH = RETRIEVAL_CFG["hnsw_ef_search"]
FAISS_CHECKPOINT_EVERY = int(RETRIEVAL_CFG.get("faiss_checkpoint_every", 10))
OLLAMA_BASE = OLLAMA_CFG["base_url"]
EMBED_MODEL = MODEL_CFG["embedding"]
EMBEDDING_SERVICE = get_embedding_service()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return EMBEDDING_SERVICE.embed_texts(texts).tolist()


def entry_text(entry: dict[str, Any]) -> str:
    return str(entry.get("text_for_embedding") or entry.get("content") or entry.get("text") or "")


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------
def load_captions() -> dict[str, str]:
    if CAPTIONS_CACHE.exists():
        with open(CAPTIONS_CACHE, encoding="utf-8") as handle:
            raw_captions = json.load(handle)
        return {Path(k).name: str(v or "") for k, v in raw_captions.items()}
    return {}


def image_filename_from_chunk(chunk: dict[str, Any]) -> str | None:
    metadata = chunk.get("metadata", {})
    image_ref = (
        metadata.get("image_filename")
        or metadata.get("image_ref")
        or metadata.get("image_path")
        or chunk.get("image_filename")
    )
    if image_ref:
        return Path(str(image_ref)).name
    match = IMAGE_NAME_RE.search(str(chunk.get("content") or chunk.get("text") or ""))
    if match:
        return Path(match.group(1).strip()).name
    return None


def get_chunk_text(chunk: dict[str, Any], captions: dict[str, str], use_captions: bool = True) -> str:
    text = str(chunk.get("content") or chunk.get("text") or "")
    if not use_captions:
        return text
    image_name = image_filename_from_chunk(chunk)
    if not image_name:
        return text
    caption = captions.get(image_name, "").strip()
    if not caption:
        return text
    return f"{text}\n\n[Figure: {caption}]"


def load_document_entries(skip_image_captions: bool = False) -> list[dict[str, Any]]:
    if not DOCUMENTS_FILE.exists():
        log.error("documents.json not found at %s - run chunker.py first", DOCUMENTS_FILE)
        sys.exit(1)

    with open(DOCUMENTS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    log.info("Loaded %d chunks from %s", len(chunks), DOCUMENTS_FILE)

    captions = {}
    if not skip_image_captions:
        captions = load_captions()
        if not captions:
            log.info("No caption cache found at %s. Continuing without caption augmentation.", CAPTIONS_CACHE)

    hits = 0
    entries: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata"), dict) else {}
        if "id" not in chunk:
            chunk_id = metadata.get("chunk_id", i)
            title = str(metadata.get("title", "unknown")).replace(" ", "_")[:30]
            chunk["id"] = f"{title}_{chunk_id}_{i}"

        text = get_chunk_text(chunk, captions, use_captions=not skip_image_captions)
        image_name = image_filename_from_chunk(chunk)
        if image_name and captions.get(image_name, "").strip():
            hits += 1

        chunk["content"] = text
        chunk["type"] = "document"
        chunk["text_for_embedding"] = text
        entries.append(chunk)

    if not skip_image_captions:
        log.info("Applied cached captions to %d document chunks", hits)
    return entries


def load_acronym_entries() -> list[dict[str, Any]]:
    if not ACRONYMS_FILE.exists():
        log.warning("acronyms.json not found at %s - skipping acronym embeddings", ACRONYMS_FILE)
        return []

    with open(ACRONYMS_FILE, encoding="utf-8") as f:
        acronyms = json.load(f)
    if not isinstance(acronyms, dict):
        log.warning("acronyms.json is not a dict - skipping acronym embeddings")
        return []

    entries: list[dict[str, Any]] = []
    for key in sorted(acronyms.keys()):
        value = str(acronyms.get(key) or "").strip()
        if not value:
            continue
        content = f"{key}: {value}"
        entries.append(
            {
                "id": f"acronym_{key}",
                "type": "acronym",
                "content": content,
                "text_for_embedding": content,
                "metadata": {
                    "chunk_id": f"acronym_{key}",
                    "chunk_type": "acronym",
                    "title": "Acronym Dictionary",
                    "source": "acronyms.json",
                    "acronym": key,
                    "expansion": value,
                    "parent_id": "acronyms",
                    "parent_title": "Acronym Dictionary",
                    "page_start": 0,
                    "page_end": 0,
                },
            }
        )

    log.info("Loaded %d acronym entries from %s", len(entries), ACRONYMS_FILE)
    return entries


def _extract_parent_summary(parent: dict[str, Any]) -> str:
    for key in ("summary", "overview", "description", "content"):
        value = parent.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_parent_entries() -> list[dict[str, Any]]:
    if not PARENT_INDEX_FILE.exists():
        log.warning("parent_index.json not found at %s - skipping parent embeddings", PARENT_INDEX_FILE)
        return []

    with open(PARENT_INDEX_FILE, encoding="utf-8") as f:
        parent_index = json.load(f)
    if not isinstance(parent_index, dict):
        log.warning("parent_index.json is not a dict - skipping parent embeddings")
        return []

    entries: list[dict[str, Any]] = []
    for parent_key in sorted(parent_index.keys()):
        item = parent_index.get(parent_key) or {}
        if not isinstance(item, dict):
            continue

        parent_id = str(item.get("parent_id") or parent_key)
        title = str(item.get("title") or parent_id)
        page_range = item.get("page_range") if isinstance(item.get("page_range"), dict) else {}
        page_start = page_range.get("start", 0)
        page_end = page_range.get("end", page_start)
        summary = _extract_parent_summary(item)

        lines = [
            f"Parent Section: {title}",
            f"Section ID: {parent_id}",
            f"Page Range: {page_start}-{page_end}",
        ]
        if summary:
            lines.append(f"Summary: {summary}")
        content = "\n".join(lines)

        entries.append(
            {
                "id": f"parent_{parent_id}",
                "type": "parent",
                "content": content,
                "text_for_embedding": content,
                "metadata": {
                    "chunk_id": f"parent_{parent_id}",
                    "chunk_type": "parent",
                    "title": title,
                    "source": "parent_index.json",
                    "parent_id": parent_id,
                    "parent_title": title,
                    "page_start": page_start,
                    "page_end": page_end,
                    "doc_count": item.get("doc_count", 0),
                    "tokens": item.get("tokens", 0),
                },
            }
        )

    log.info("Loaded %d parent entries from %s", len(entries), PARENT_INDEX_FILE)
    return entries


def load_image_caption_entries() -> list[dict[str, Any]]:
    captions = load_captions()
    if not captions:
        log.info("No image caption entries loaded from %s", CAPTIONS_CACHE)
        return []

    entries: list[dict[str, Any]] = []
    for idx, image_name in enumerate(sorted(captions.keys())):
        caption = str(captions.get(image_name) or "").strip()
        if not caption:
            continue

        content = f"Image: {image_name}\nCaption: {caption}"
        entries.append(
            {
                "id": f"image_caption_{idx}",
                "type": "image_caption",
                "content": content,
                "text_for_embedding": content,
                "metadata": {
                    "chunk_id": f"image_caption_{idx}",
                    "chunk_type": "image_caption",
                    "title": "Image Captions",
                    "source": "image_captions.json",
                    "image_filename": image_name,
                    "page_start": 0,
                    "page_end": 0,
                },
            }
        )

    log.info("Loaded %d image caption entries from %s", len(entries), CAPTIONS_CACHE)
    return entries


def load_corpus(skip_image_captions: bool = False) -> list[dict[str, Any]]:
    documents = load_document_entries(skip_image_captions=skip_image_captions)
    acronyms = load_acronym_entries()
    parents = load_parent_entries()
    image_captions = load_image_caption_entries()
    corpus = documents + acronyms + parents + image_captions

    if not corpus:
        log.error("No corpus entries found. Nothing to index.")
        sys.exit(1)

    log.info(
        "Unified corpus size: %d (documents=%d, acronyms=%d, parents=%d, image_captions=%d)",
        len(corpus),
        len(documents),
        len(acronyms),
        len(parents),
        len(image_captions),
    )
    return corpus


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def _persist_faiss_files(index: faiss.Index, id_map: dict[str, dict[str, Any]]) -> None:
    FAISS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX))
    with open(FAISS_ID_MAP, "w", encoding="utf-8") as handle:
        json.dump(id_map, handle, indent=2, ensure_ascii=False)
    with open(FAISS_CORPUS_JSON, "w", encoding="utf-8") as handle:
        json.dump({"entries": list(id_map.values())}, handle, indent=2, ensure_ascii=False)


def index_faiss(corpus: list[dict[str, Any]], reset: bool) -> None:
    if reset:
        for path in (FAISS_INDEX, FAISS_ID_MAP, FAISS_CORPUS_JSON):
            if path.exists():
                path.unlink()

    index = faiss.IndexHNSWFlat(EMBEDDING_SERVICE.dimension, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH

    id_map: dict[str, dict[str, Any]] = {}
    vectors: list[np.ndarray] = []

    log.info(
        "Embedding and indexing %d entries into FAISS with %s (%d dims) ...",
        len(corpus),
        EMBEDDING_SERVICE.provider,
        EMBEDDING_SERVICE.dimension,
    )

    for idx, entry in enumerate(corpus):
        text = entry_text(entry)

        try:
            vector = EMBEDDING_SERVICE.embed_texts([text])[0]
        except Exception:
            cid = entry.get("id", idx)
            log.error("Embed failed at entry index %s id=%r - first 200 chars: %r", idx, cid, text[:200])
            raise

        vectors.append(vector)
        id_map[str(idx)] = entry

        batch_full = len(vectors) >= BATCH_SIZE
        checkpoint_hit = FAISS_CHECKPOINT_EVERY > 0 and (idx + 1) % FAISS_CHECKPOINT_EVERY == 0 and vectors
        if batch_full or checkpoint_hit:
            matrix = np.stack(vectors).astype(np.float32)
            index.add(matrix)
            vectors.clear()
            log.info("Indexed %d/%d entries", idx + 1, len(corpus))
            if checkpoint_hit:
                _persist_faiss_files(index, id_map)
                log.info("FAISS checkpoint: %d vectors written to disk", index.ntotal)

    if vectors:
        matrix = np.stack(vectors).astype(np.float32)
        index.add(matrix)
        vectors.clear()
        log.info("Indexed %d/%d entries", len(corpus), len(corpus))

    _persist_faiss_files(index, id_map)

    counts: dict[str, int] = {}
    for entry in id_map.values():
        t = str(entry.get("type") or "document")
        counts[t] = counts.get(t, 0) + 1

    log.info("FAISS: %d entries added to vector index", int(index.ntotal))
    log.info("FAISS type distribution: %s", counts)
    log.info("FAISS index saved to %s", FAISS_INDEX)
    log.info("FAISS id map saved to %s (%d entries)", FAISS_ID_MAP, len(id_map))
    log.info("Unified corpus JSON saved to %s", FAISS_CORPUS_JSON)


def index_bm25(corpus: list[dict[str, Any]], reset: bool) -> None:
    if reset and BM25_FILE.exists():
        BM25_FILE.unlink()

    log.info("Building BM25 index over %d entries ...", len(corpus))
    tokenized = [entry_text(entry).lower().split() for entry in corpus]
    bm25 = BM25Okapi(tokenized)

    BM25_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": corpus}, f)

    log.info("BM25 index saved to %s", BM25_FILE)


def smoke_test() -> None:
    test_query = "What are the phases of the NASA project life cycle?"
    log.info("Smoke test query: '%s'", test_query)

    index = faiss.read_index(str(FAISS_INDEX))
    index.hnsw.efSearch = HNSW_EF_SEARCH
    with open(FAISS_ID_MAP, encoding="utf-8") as handle:
        id_map = json.load(handle)

    query_embedding = EMBEDDING_SERVICE.embed_texts([test_query], is_query=True)
    scores, indices = index.search(query_embedding, k=5)

    log.info("Top results:")
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = id_map.get(str(idx), {})
        metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
        etype = str(entry.get("type") or "document")
        title = metadata.get("title", "?")
        page_start = metadata.get("page_start", "?")
        page_end = metadata.get("page_end", "?")
        preview = entry_text(entry).replace("\n", " ")[:100]
        log.info("  [score=%.4f] type=%s %s (p%s-%s) | %s...", score, etype, title, page_start, page_end, preview)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FAISS + BM25 indexes for NASA handbook RAG")
    p.add_argument("--reset", action="store_true", help="Wipe existing indexes and rebuild")
    p.add_argument("--skip-image-captions", action="store_true", help="Skip caption augmentation of image chunks")
    p.add_argument("--smoke-test", action="store_true", default=True, help="Run smoke test after indexing")
    p.add_argument("--no-smoke-test", dest="smoke_test", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=== NASA Handbook Indexer (FAISS + BM25) ===")
    log.info("Reset=%s  SkipCaptions=%s", args.reset, args.skip_image_captions)
    log.info("FAISS index path (absolute): %s", FAISS_INDEX.resolve())
    log.info("FAISS id map path (absolute): %s", FAISS_ID_MAP.resolve())

    corpus = load_corpus(skip_image_captions=args.skip_image_captions)
    index_faiss(corpus, reset=args.reset)
    index_bm25(corpus, reset=args.reset)

    if args.smoke_test:
        smoke_test()

    log.info("=== Indexing complete ===")
    log.info("  FAISS Index   -> %s", FAISS_INDEX)
    log.info("  FAISS ID Map  -> %s", FAISS_ID_MAP)
    log.info("  Corpus JSON   -> %s", FAISS_CORPUS_JSON)
    log.info("  BM25 Index    -> %s", BM25_FILE)
    if not args.skip_image_captions:
        log.info("  Captions      -> %s", CAPTIONS_CACHE)


if __name__ == "__main__":
    main()