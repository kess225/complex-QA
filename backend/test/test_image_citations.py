"""Validate that image chunks in FAISS id map include cached figure captions."""

import json
from pathlib import Path

import pytest
import yaml


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _load_faiss_id_map_path() -> Path:
    settings_path = BACKEND_ROOT / "settings.yaml"
    with open(settings_path, encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)

    configured_path = settings["retrieval"]["faiss_id_map_path"]
    path = Path(configured_path)
    return path if path.is_absolute() else BACKEND_ROOT / path


def _iter_chunks(id_map_data: object) -> list[dict]:
    if isinstance(id_map_data, dict):
        return [chunk for chunk in id_map_data.values() if isinstance(chunk, dict)]
    if isinstance(id_map_data, list):
        return [chunk for chunk in id_map_data if isinstance(chunk, dict)]
    raise TypeError("FAISS id map must be a dict or a list of chunk objects")


def _is_image_chunk(chunk: dict) -> bool:
    metadata = chunk.get("metadata", {})
    content = str(chunk.get("content", ""))

    if not isinstance(metadata, dict):
        metadata = {}

    image_markers = (
        metadata.get("chunk_type") == "image",
        bool(metadata.get("image_filename")),
        bool(metadata.get("image_ref")),
        bool(metadata.get("image_path")),
        bool(chunk.get("image_filename")),
        "[Image:" in content,
    )
    return any(image_markers)


def test_image_chunks_include_figure_citations() -> None:
    """Ensure caption augmentation made it into persisted FAISS chunk content."""
    id_map_path = _load_faiss_id_map_path()

    if not id_map_path.exists():
        pytest.skip(
            f"FAISS id map not found at {id_map_path}; run backend/ingestion/indexer.py first."
        )

    with open(id_map_path, encoding="utf-8") as handle:
        id_map_data = json.load(handle)

    chunks = _iter_chunks(id_map_data)
    if not chunks:
        pytest.skip("FAISS id map is empty; no chunks to validate.")

    image_chunks = [chunk for chunk in chunks if _is_image_chunk(chunk)]
    if not image_chunks:
        pytest.skip("No image chunks found in FAISS id map.")

    missing_citations = [
        chunk.get("id") or chunk.get("metadata", {}).get("chunk_id") or "unknown"
        for chunk in image_chunks
        if "[Figure:" not in str(chunk.get("content", ""))
    ]

    assert not missing_citations, (
        "Image chunks missing [Figure:] caption text in content: "
        f"{missing_citations[:10]}"
    )