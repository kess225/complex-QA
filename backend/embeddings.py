import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import yaml

log = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent
SETTINGS_FILE = BACKEND_ROOT / "settings.yaml"
_EMBED_EMPTY_PLACEHOLDER = "[empty chunk]"
_DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    with open(SETTINGS_FILE, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class EmbeddingService:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.models_cfg = config.get("models", {})
        self.embeddings_cfg = config.get("embeddings", {})
        self.provider_cfg = self.embeddings_cfg.get("providers", {})
        self.ollama_cfg = config.get("ollama", {})

        self.provider = str(self.embeddings_cfg.get("provider", "huggingface")).strip().lower()
        self.dimension = int(self.embeddings_cfg.get("dimension", 384))
        self.normalize = bool(self.embeddings_cfg.get("normalize", True))
        self.query_prefix = str(self.embeddings_cfg.get("query_prefix", _DEFAULT_QUERY_PREFIX))

        huggingface_cfg = self.provider_cfg.get("huggingface", {})
        ollama_provider_cfg = self.provider_cfg.get("ollama", {})

        self.huggingface_model = str(
            huggingface_cfg.get("model_name")
            or self.models_cfg.get("embedding")
            or "BAAI/bge-small-en-v1.5"
        )
        self.huggingface_device = str(huggingface_cfg.get("device", "cpu"))
        self.huggingface_trust_remote_code = bool(huggingface_cfg.get("trust_remote_code", False))

        self.ollama_model = str(
            ollama_provider_cfg.get("model_name")
            or self.models_cfg.get("embedding")
            or "nomic-embed-text"
        )
        self.ollama_base_url = str(self.ollama_cfg.get("base_url", "http://localhost:11434"))
        self.ollama_timeout = int(self.ollama_cfg.get("timeout", 60))
        self.ollama_max_embed_chars = int(self.ollama_cfg.get("max_embed_chars", 12000))

        self._model = None
        self._model_lock = threading.Lock()

    def _prepare_text(self, text: str, *, is_query: bool) -> str:
        value = str(text).encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")
        value = value.replace("\x00", "").strip()
        if not value:
            value = _EMBED_EMPTY_PLACEHOLDER
        if self.provider == "ollama" and len(value) > self.ollama_max_embed_chars:
            value = value[: self.ollama_max_embed_chars]
        if is_query and self.provider == "huggingface" and self.query_prefix:
            value = f"{self.query_prefix}{value}"
        return value

    def _ensure_dimension(self, actual_dimension: int) -> None:
        if actual_dimension != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {actual_dimension} "
                f"from provider '{self.provider}'"
            )

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if not self.normalize:
            self._ensure_dimension(int(matrix.shape[1]))
            return matrix

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        matrix = matrix / norms
        self._ensure_dimension(int(matrix.shape[1]))
        return matrix.astype(np.float32)

    def _get_huggingface_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:
                        raise ImportError(
                            "sentence-transformers is required for Hugging Face embeddings. "
                            "Run 'uv sync' after updating dependencies."
                        ) from exc

                    self._model = SentenceTransformer(
                        self.huggingface_model,
                        device=self.huggingface_device,
                        trust_remote_code=self.huggingface_trust_remote_code,
                    )
        return self._model

    def _embed_with_huggingface(self, texts: list[str]) -> np.ndarray:
        model = self._get_huggingface_model()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return self._normalize_vectors(np.asarray(vectors, dtype=np.float32))

    def _embed_with_ollama(self, texts: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for payload in texts:
            response = httpx.post(
                f"{self.ollama_base_url}/api/embed",
                json={"model": self.ollama_model, "input": payload, "truncate": True},
                timeout=self.ollama_timeout,
            )

            body_lower = (response.text or "").lower()
            if response.status_code == 400 and "context length" in body_lower and len(payload) > 2048:
                for cap in (4096, 2048):
                    response = httpx.post(
                        f"{self.ollama_base_url}/api/embed",
                        json={"model": self.ollama_model, "input": payload[:cap], "truncate": True},
                        timeout=self.ollama_timeout,
                    )
                    if response.is_success:
                        break
                    body_lower = (response.text or "").lower()
                    if response.status_code != 400 or "context length" not in body_lower:
                        break

            if not response.is_success:
                log.error(
                    "Ollama /api/embed HTTP %s - body: %s",
                    response.status_code,
                    (response.text or "")[:2000],
                )
            response.raise_for_status()

            embeddings = response.json().get("embeddings") or []
            if not embeddings:
                raise ValueError("Ollama returned no embeddings; check model name and Ollama version")

            rows.append(np.asarray(embeddings[0], dtype=np.float32))

        return self._normalize_vectors(np.stack(rows))

    def embed_texts(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        prepared = [self._prepare_text(text, is_query=is_query) for text in texts]
        if self.provider == "huggingface":
            return self._embed_with_huggingface(prepared)
        if self.provider == "ollama":
            return self._embed_with_ollama(prepared)
        raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text], is_query=True)[0]


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(load_settings())