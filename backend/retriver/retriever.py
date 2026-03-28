"""
retriever.py
-----------
Hybrid retriever that combines FAISS dense search + BM25 sparse search
over a unified typed corpus (document, acronym, parent).
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi
import yaml

from backend.embeddings import get_embedding_service

log = logging.getLogger(__name__)

# ====================================================================
# CONFIG
# ====================================================================
BACKEND_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = BACKEND_ROOT / "settings.yaml"


def resolve_repo_path(path_value: str | Path) -> Path:
	path = Path(path_value)
	return path if path.is_absolute() else BACKEND_ROOT / path


if SETTINGS_FILE.exists():
	with open(SETTINGS_FILE, encoding="utf-8") as f:
		CFG = yaml.safe_load(f) or {}
else:
	CFG = {}

RETRIEVAL_CFG = CFG.get("retrieval", {})
OLLAMA_CFG = CFG.get("ollama", {})
MODEL_CFG = CFG.get("models", {})

FAISS_INDEX = resolve_repo_path(RETRIEVAL_CFG.get("faiss_index_path", "db/faiss_store/index.faiss"))
FAISS_ID_MAP = resolve_repo_path(RETRIEVAL_CFG.get("faiss_id_map_path", "db/faiss_store/id_map.json"))
BM25_FILE = BACKEND_ROOT / "db" / "bm25_index.pkl"
CAPTIONS_CACHE = BACKEND_ROOT / "extracted" / "image_captions.json"

HNSW_EF_SEARCH = int(RETRIEVAL_CFG.get("hnsw_ef_search", 100))
RRF_K = int(RETRIEVAL_CFG.get("rrf_k", 60))
TOP_K_DENSE = int(RETRIEVAL_CFG.get("dense_top_k", 10))
TOP_K_SPARSE = int(RETRIEVAL_CFG.get("sparse_top_k", 10))
MAX_CANDIDATES = 20
DEDUP_COSINE_THRESHOLD = float(RETRIEVAL_CFG.get("dedup_threshold", 0.92))

OLLAMA_BASE = OLLAMA_CFG.get("base_url", "http://localhost:11434")
EMBEDDING_SERVICE = get_embedding_service()

IMAGE_NAME_RE = re.compile(r"\[Image:\s*([^\]]+)\]")


# ====================================================================
# HELPERS
# ====================================================================
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
	norm1 = np.linalg.norm(vec1)
	norm2 = np.linalg.norm(vec2)
	if norm1 == 0 or norm2 == 0:
		return 0.0
	return float(np.dot(vec1, vec2) / (norm1 * norm2))


def load_captions() -> dict[str, str]:
	if CAPTIONS_CACHE.exists():
		try:
			with open(CAPTIONS_CACHE, encoding="utf-8") as f:
				raw_captions = json.load(f)
			return {Path(key).name: str(value or "") for key, value in raw_captions.items()}
		except Exception as e:
			log.warning("Could not load captions: %s", e)
			return {}
	return {}


def extract_image_name(content: str) -> str | None:
	match = IMAGE_NAME_RE.search(content)
	if match:
		return Path(match.group(1).strip()).name
	return None


def augment_with_caption(content: str, captions: dict[str, str]) -> str:
	image_name = extract_image_name(content)
	if not image_name:
		return content

	caption = captions.get(image_name, "").strip()
	if not caption:
		return content

	return IMAGE_NAME_RE.sub(lambda m: f"{m.group(0)}\nCaption: {caption}", content)


def entry_text(entry: dict[str, Any]) -> str:
	return str(entry.get("text_for_embedding") or entry.get("content") or entry.get("text") or "")


def normalize_id_map(raw: Any) -> dict[str, dict[str, Any]]:
	if isinstance(raw, dict):
		out: dict[str, dict[str, Any]] = {}
		for k, v in raw.items():
			if isinstance(v, dict):
				out[str(k)] = v
		return out
	if isinstance(raw, list):
		out = {}
		for i, value in enumerate(raw):
			if isinstance(value, dict):
				out[str(i)] = value
		return out
	return {}


# ====================================================================
# HYBRID RETRIEVER
# ====================================================================
class HybridRetriever(BaseRetriever):
	model_config = ConfigDict(arbitrary_types_allowed=True)

	top_k_dense: int = TOP_K_DENSE
	top_k_sparse: int = TOP_K_SPARSE
	rrf_k: int = RRF_K
	dedup_threshold: float = DEDUP_COSINE_THRESHOLD
	max_candidates: int = MAX_CANDIDATES
	use_captions: bool = True

	faiss_index: Any = None
	id_map: dict[str, dict[str, Any]] = {}
	bm25: Any = None
	corpus: list[dict[str, Any]] = []
	captions: dict[str, str] = {}

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._load_indices()
		self._load_captions()

	def _load_indices(self) -> None:
		if not FAISS_INDEX.exists():
			raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX}")
		if not FAISS_ID_MAP.exists():
			raise FileNotFoundError(f"FAISS ID map not found: {FAISS_ID_MAP}")

		self.faiss_index = faiss.read_index(str(FAISS_INDEX))
		self.faiss_index.hnsw.efSearch = HNSW_EF_SEARCH
		if int(getattr(self.faiss_index, "d", -1)) != EMBEDDING_SERVICE.dimension:
			raise ValueError(
				f"FAISS index dimension {getattr(self.faiss_index, 'd', None)} does not match "
				f"configured embedding dimension {EMBEDDING_SERVICE.dimension}. Rebuild the index."
			)
		log.info("FAISS loaded: %d vectors", self.faiss_index.ntotal)

		with open(FAISS_ID_MAP, encoding="utf-8") as f:
			self.id_map = normalize_id_map(json.load(f))
		log.info("ID map loaded: %d entries", len(self.id_map))

		if not BM25_FILE.exists():
			raise FileNotFoundError(f"BM25 index not found: {BM25_FILE}")

		with open(BM25_FILE, "rb") as f:
			data = pickle.load(f)
			self.bm25 = data["bm25"]
			self.corpus = data.get("chunks", [])
		log.info("BM25 loaded: %d entries", len(self.corpus))

	def _load_captions(self) -> None:
		self.captions = load_captions() if self.use_captions else {}
		if self.captions:
			log.info("Captions loaded: %d images", len(self.captions))

	def _embed_query(self, query: str) -> np.ndarray:
		return EMBEDDING_SERVICE.embed_query(query)

	def _dense_search(self, query_vector: np.ndarray) -> dict[int, float]:
		query_vector = np.array([query_vector], dtype=np.float32)
		scores, indices = self.faiss_index.search(query_vector, self.top_k_dense)
		return {int(idx): float(score) for idx, score in zip(indices[0], scores[0]) if idx >= 0}

	def _sparse_search(self, query_text: str) -> dict[int, float]:
		tokens = query_text.lower().split()
		scores = self.bm25.get_scores(tokens)
		top_indices = np.argsort(scores)[::-1][: self.top_k_sparse]
		return {int(idx): float(scores[idx]) for idx in top_indices if scores[idx] > 0}

	def _rrf_fusion(self, dense_results: dict[int, float], sparse_results: dict[int, float]) -> dict[int, dict[str, Any]]:
		dense_ranks = {idx: rank for rank, idx in enumerate(dense_results.keys(), 1)}
		sparse_ranks = {idx: rank for rank, idx in enumerate(sparse_results.keys(), 1)}

		all_indices = set(dense_ranks.keys()) | set(sparse_ranks.keys())
		rrf_scores: dict[int, dict[str, Any]] = {}

		for idx in all_indices:
			score = 0.0
			if idx in dense_ranks:
				score += 1.0 / (self.rrf_k + dense_ranks[idx])
			if idx in sparse_ranks:
				score += 1.0 / (self.rrf_k + sparse_ranks[idx])
			rrf_scores[idx] = {
				"rrf_score": score,
				"dense_rank": dense_ranks.get(idx),
				"sparse_rank": sparse_ranks.get(idx),
			}

		return dict(sorted(rrf_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True))

	def _dedup_candidates(self, rrf_results: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
		sorted_results = sorted(rrf_results.items(), key=lambda x: x[1]["rrf_score"], reverse=True)
		to_keep = set()
		vector_cache: dict[int, np.ndarray] = {}

		for idx, _data in sorted_results:
			entry = self.id_map.get(str(idx), {})
			if str(entry.get("type") or "document") != "document":
				to_keep.add(idx)
				continue

			try:
				vector = self.faiss_index.reconstruct(idx)
			except Exception as e:
				log.warning("Cannot reconstruct vector %d: %s", idx, e)
				to_keep.add(idx)
				continue

			is_duplicate = False
			for kept_idx in to_keep:
				kept_entry = self.id_map.get(str(kept_idx), {})
				if str(kept_entry.get("type") or "document") != "document":
					continue

				if kept_idx not in vector_cache:
					try:
						vector_cache[kept_idx] = self.faiss_index.reconstruct(kept_idx)
					except Exception:
						continue

				if cosine_similarity(vector, vector_cache[kept_idx]) > self.dedup_threshold:
					is_duplicate = True
					break

			if not is_duplicate:
				to_keep.add(idx)
				vector_cache[idx] = vector

		return {idx: data for idx, data in rrf_results.items() if idx in to_keep}

	def _build_document_from_entry(self, idx: int, entry: dict[str, Any], score_data: dict[str, Any]) -> Document:
		entry_type = str(entry.get("type") or "document")
		metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}

		content = str(entry.get("content") or entry_text(entry))
		if entry_type == "document" and self.use_captions and self.captions:
			content = augment_with_caption(content, self.captions)

		source = metadata.get("title") or metadata.get("source") or "unknown"
		if entry_type == "acronym":
			source = "Acronym Dictionary"
		elif entry_type == "parent":
			source = metadata.get("title") or "Parent Section"

		return Document(
			page_content=content,
			metadata={
				"chunk_id": metadata.get("chunk_id", entry.get("id", f"chunk_{idx}")),
				"entry_type": entry_type,
				"source": source,
				"page": metadata.get("page_start", 0),
				"page_end": metadata.get("page_end", 0),
				"parent_id": metadata.get("parent_id"),
				"parent_title": metadata.get("parent_title") or metadata.get("title"),
				"rrf_score": score_data["rrf_score"],
				"dense_rank": score_data.get("dense_rank"),
				"sparse_rank": score_data.get("sparse_rank"),
			},
		)

	def _build_documents(self, rrf_results: dict[int, dict[str, Any]]) -> list[Document]:
		documents: list[Document] = []
		for idx in list(rrf_results.keys())[: self.max_candidates]:
			entry = self.id_map.get(str(idx))
			if not entry:
				continue
			documents.append(self._build_document_from_entry(idx, entry, rrf_results[idx]))
		return documents

	def _get_relevant_documents(self, query: str) -> list[Document]:
		dense_results: dict[int, float] = {}
		try:
			query_embedding = self._embed_query(query)
			dense_results = self._dense_search(query_embedding)
		except Exception as e:
			log.warning("Dense embedding failed (%s); continuing with sparse retrieval only", e)

		sparse_results = self._sparse_search(query)
		rrf_results = self._rrf_fusion(dense_results, sparse_results)
		dedup_results = self._dedup_candidates(rrf_results)
		return self._build_documents(dedup_results)

	def invoke(self, input: str, **kwargs) -> list[Document]:
		return self._get_relevant_documents(input)

	def batch(self, inputs: list[str], **kwargs) -> list[list[Document]]:
		return [self._get_relevant_documents(q) for q in inputs]

