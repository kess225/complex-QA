# NASA RAG App

This project includes:

- A **FastAPI** backend for query execution (`backend/main.py`)
- An optional **CLI** for ad-hoc questions without the API (`backend/cli.py`)
- A **Streamlit** frontend that calls the backend API

Configuration, raw data, FAISS/BM25 indexes, extracted JSON, and tests live under **`backend/`**. Environment variables belong in **`.env`** at the **repository root** (start from `.env.example`). Details: [ARCHITECTURE.md](ARCHITECTURE.md).

## Project layout

```
rag/
├── pyproject.toml
├── uv.lock
├── .gitignore
├── README.md
├── ARCHITECTURE.md
├── .env.example
├── frontend/
│   ├── app.py                 ← Streamlit UI
│   └── README.md
└── backend/
    ├── settings.yaml
    ├── main.py                ← FastAPI application
    ├── cli.py                 ← CLI: python -m backend.cli
    ├── embeddings.py
    ├── api/
    │   ├── __init__.py
    │   ├── dependencies.py
    │   ├── query.py           ← POST /api/query
    │   └── schemas.py
    ├── retriever/
    │   ├── retriever.py
    │   ├── context_builder.py
    │   └── rag_chain.py
    ├── ingestion/
    │   ├── chunker.py
    │   ├── indexer.py
    │   ├── verify_index.py
    │   ├── image_captioner.py
    │   ├── acronym_extractor.py
    │   └── table_extractor.py
    ├── extracted/             ← documents.json, parent_index.json, acronyms.json, …
    ├── data/raw/              ← PDF, rag_structure_csv.csv
    ├── db/                    ← faiss_store/, bm25_index.pkl
    ├── test/                  ← pytest (configured in pyproject.toml)
    ├── backup/
    └── Intstructions/         ← internal notes (spelling as in repo)
```

## Run

1. Install dependencies:

```bash
uv sync
```

2. Start the FastAPI backend:

```bash
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

3. Start the Streamlit frontend (new terminal):

```bash
uv run streamlit run frontend/app.py
```

4. Open:

- API docs: http://localhost:8000/docs
- Frontend: http://localhost:8501

### CLI (optional)

From the repository root, run a one-off query against the same RAG chain as the API:

```bash
uv run python -m backend.cli "What are the entry criteria for PDR?"
```

Optional: `--memory "..."` for conversation context.

### Tests

```bash
uv run pytest
```

Tests are discovered under `backend/test/` (see `pyproject.toml`).

## API Contract

### POST /api/query

Request:

```json
{
	"query": "What is the NASA risk management process?",
	"memory_str": "Optional conversation memory"
}
```

Response:

```json
{
	"answer": "...",
	"documents": [
		{
			"page_content": "...",
			"metadata": {
				"chunk_id": "...",
				"entry_type": "document",
				"source": "...",
				"page": 1,
				"page_end": 2,
				"parent_id": "...",
				"parent_title": "...",
				"rrf_score": 0.1,
				"dense_rank": 1,
				"sparse_rank": 3
			}
		}
	]
}
```
