# NASA Systems Engineering Handbook — RAG System
### Hireathon 2026 · Problem Statement 2 · Complete Architecture

---

## Document Coverage

### 6 Main Chapters (Primary QA Content)

| # | Chapter | PDF Pages | Book Pages |
|---|---|---|---|
| 1 | 1.0 Introduction | 14 – 15 | 1 – 2 |
| 2 | 2.0 Fundamentals of Systems Engineering | 16 – 31 | 3 – 18 |
| 3 | 3.0 NASA Program/Project Life Cycle | 32 – 62 | 19 – 49 |
| 4 | 4.0 System Design Processes | 63 – 101 | 50 – 88 |
| 5 | 5.0 Product Realization | 102 – 143 | 89 – 130 |
| 6 | 6.0 Crosscutting Technical Management | 144 – 214 | 131 – 201 |

Total chapter pages: **201 PDF pages**

### Included Appendices (Substantive Content)

| Appendix | Title | PDF Pages | Reason Kept |
|---|---|---|---|
| A | Acronyms | 215 – 219 | Acronym dictionary extraction only |
| B | Glossary | 220 – 248 | 29 pages of term definitions |
| C | How to Write a Good Requirement | 249 – 252 | Actionable checklist |
| G | Technology Assessment / Insertion | 259 – 268 | Substantive 10-page content |
| I | Verification and Validation Plan Outline | 272 – 281 | Substantive 10-page content |
| J | SEMP Content Outline | 282 – 296 | Substantive 15-page content |
| T | Systems Engineering in Phase E | 305 – 324 | Substantive 20-page content |

### Skipped Sections

| Section | Reason |
|---|---|
| TOC, Table of Figures, Table of Tables, Table of Blue Boxes | Index pages — redundant with extraction metadata |
| Preface, Acknowledgments | Explicitly marked "Not to be considered" |
| Appendix D, E, F, H, K, L, M, N, P, R, S | Template/outline/reference only — no answerable content |
| References Cited, Bibliography | Citation lists — no QA content |

---

## Technology Stack

| Layer | Technology | Details |
|---|---|---|
| Frontend | Streamlit | localhost:8501 |
| API | FastAPI | localhost:8000 · **REST** (`GET /health`, `POST /api/query`) — no WebSocket in this repo |
| LLM (optional) | Ollama | localhost:11434 · used when `models.llm.provider` is `ollama` in `settings.yaml` |
| Primary LLM (Ollama path) | qwen2.5:7b (example) | configurable under `ollama` / `models` |
| Chat LLM (default path) | Groq | configured via `groq` + `GROQ_API_KEY` · see `backend/settings.yaml` |
| Text Embeddings | BAAI/bge-small-en-v1.5 | local Hugging Face · 384-dim · query + chunk embedding |
| Image Captioning | Gemini 2.0 Flash | API service · index time only · ~200 images |
| Re-ranker (planned / not wired) | cross-encoder | not present under `backend/retriever/` in this tree |
| Vector Store | FAISS (persistent) | `backend/db/faiss_store/` · IndexHNSWFlat · see `settings.yaml` |
| Sparse Index | BM25 — rank_bm25 | `backend/db/bm25_index.pkl` |
| Sessions / SQLite | — | not part of the current repository layout |
| Package Manager | uv | pyproject.toml single source of truth |
| IDE | Cursor | AI-assisted development |

---

## Embedding Strategy

### Hybrid approach — local query path + API-augmented index

| Stage | Model | When | Why |
|---|---|---|---|
| Text chunk embedding | BAAI/bge-small-en-v1.5 (Hugging Face) | Index time | Local · 384-dim · shared provider layer |
| Image captioning | Gemini 2.0 Flash API | Index time only · one-time | Converts figures/diagrams to searchable text via API |
| Caption embedding | BAAI/bge-small-en-v1.5 (Hugging Face) | Index time only | Caption appended to chunk text before embed |
| Query embedding | BAAI/bge-small-en-v1.5 (Hugging Face) | Every query | Local · 384-dim · query prefix tuned for retrieval |

### Image captioning flow (index time only)

```
backend/extracted/images/*.png
        │
        ▼
Gemini 2.0 Flash API  (Vision model)
  prompt: "Describe this technical diagram precisely.
           Include all labels, axis names, process steps,
           and relationships shown."
        │
        ▼
caption string  (e.g. "V-model diagram showing seven phases
                 across Problem Space and Solution Space axes...")
        │
        ▼
appended to parent chunk text
        │
        ▼
BAAI/bge-small-en-v1.5  →  384-dim L2-normalised vector
        │
        ▼
FAISS IndexHNSWFlat
```

**Result:** image content is now searchable via text queries.
No API call occurs at query time — Gemini is only used during ingestion (e.g. `image_captioner.py` / indexer pipeline), not per query.

---

## Full System Architecture

The ASCII diagram below describes a **richer target** (sessions, eval dashboard, WebSocket, extra LLM tiers). **As implemented today**, the runnable path is: **Streamlit** → **FastAPI REST** (`/api/query`) → **HybridRetriever** + **RAGChain** (Groq or Ollama). See [File Structure](#file-structure) for the exact tree.

```
╔══════════════════════════════════════════════════════════════════════════╗
║                              FRONTEND                                   ║
║                      Streamlit  localhost:8501                          ║
║                                                                         ║
║   ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ ║
║   │   Chat UI   │  │ Source Panel │  │    Eval      │  │  Session   │ ║
║   │             │  │              │  │  Dashboard   │  │  Selector  │ ║
║   │ Streaming   │  │ Collapsible  │  │              │  │            │ ║
║   │ response    │  │ citations    │  │ Score badges │  │ History    │ ║
║   │             │  │ page + sec   │  │ recall       │  │ navigation │ ║
║   │             │  │ ref shown    │  │ confidence   │  │            │ ║
║   │             │  │              │  │ correctness  │  │            │ ║
║   │             │  │              │  │ groundedness │  │            │ ║
║   └─────────────┘  └──────────────┘  └──────────────┘  └────────────┘ ║
╚══════════════════════════════════════╦══════════════════════════════════╝
                                       ║ HTTP REST / WebSocket (streaming)
╔══════════════════════════════════════╩══════════════════════════════════╗
║                            API GATEWAY                                 ║
║                      FastAPI  localhost:8000                            ║
║                                                                         ║
║   POST /query          POST /ingest           GET /health               ║
║   WS   /query/stream   GET  /sessions         GET /health/models        ║
║                                                                         ║
║   Pydantic schema validation  ·  CORS middleware  ·  Request-ID inject  ║
╚═══════════╦═══════════════════════════╦══════════════════╦═════════════╝
            ║                           ║                  ║
╔═══════════╩════════╗   ╔═════════════╩══════╗   ╔═══════╩════════════╗
║   QUERY PROCESSOR  ║   ║   MEMORY MANAGER   ║   ║     LLM ROUTER     ║
║                    ║   ║                    ║   ║                    ║
║  1. Acronym expand ║   ║  Tier 1            ║   ║  Classify intent   ║
║     acronyms.json  ║   ║  Short-term buffer ║   ║                    ║
║                    ║   ║  Last 6 turns      ║   ║  fast_intents      ║
║  2. Intent classif ║   ║  verbatim in prompt║   ║  conversational    ║
║     factual        ║   ║                    ║   ║  factual           ║
║     multi_hop      ║   ║  Tier 2            ║   ║  summarise         ║
║     summarise      ║   ║  Rolling summary   ║   ║  → qwen2:1.5b      ║
║     conversational ║   ║  Compressed hist   ║   ║                    ║
║                    ║   ║  qwen2:1.5b model  ║   ║  deep_intents      ║
║  3. Query rewrite  ║   ║  every 6 turns     ║   ║  multi_hop         ║
║     optimised for  ║   ║                    ║   ║  cross_chapter     ║
║     retrieval      ║   ║  Tier 3            ║   ║  comparative       ║
║                    ║   ║  Entity store      ║   ║  → qwen2.5:7b      ║
║                    ║   ║  Sections seen     ║   ║                    ║
║                    ║   ║  Topics tracked    ║   ║  Health monitor    ║
║                    ║   ║                    ║   ║  Timeout fallback  ║
║                    ║   ║  Token budget gate ║   ║  → qwen2:1.5b      ║
║                    ║   ║  Max 3800 tokens   ║   ║                    ║
║                    ║   ║  SQLite persist    ║   ║  Latency logging   ║
╚═══════════╦════════╝   ╚═════════════╦══════╝   ╚═══════╦════════════╝
            ╚═══════════════════╦═══════╝                  ║
                                ╚══════════════════════════╝
                                             ║
╔════════════════════════════════════════════╩════════════════════════════╗
║                            RAG PIPELINE                                ║
║                                                                        ║
║  ┌──────────────────────────────────────────────────────────────────┐  ║
║  │                         RETRIEVER                               │  ║
║  │                                                                  │  ║
║  │   Dense Search                    Sparse Search                 │  ║
║  │   FAISS IndexHNSWFlat             BM25 (rank_bm25)              │  ║
║  │   BAAI/bge-small-en-v1.5          Okapi BM25                    │  ║
║  │   384-dim cosine (IP+L2norm)      keyword matching              │  ║
║  │   M=32 · efSearch=100             lower-cased word tokens       │  ║
║  │   top-10 candidates               top-10 candidates             │  ║
║  │              │                              │                   │  ║
║  │              └──────────┬───────────────────┘                   │  ║
║  │                         ▼                                        │  ║
║  │        RRF Fusion  k=60 — implemented in HybridRetriever        │  ║
║  │      Reciprocal Rank Fusion merge → fused candidate set         │  ║
║  │    Dedup filter: cosine sim > configurable threshold → drop     │  ║
║  └──────────────────────────┬───────────────────────────────────────┘  ║
║                             ▼                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐  ║
║  │                        RE-RANKER                                │  ║
║  │                                                                  │  ║
║  │   cross-encoder/ms-marco-MiniLM-L-6-v2                          │  ║
║  │   Local HuggingFace · runs on CPU                               │  ║
║  │   Joint query+document scoring                                  │  ║
║  │   ~200ms for 10 candidates                                      │  ║
║  │   Output: final top-3  +  confidence score → eval layer         │  ║
║  └──────────────────────────┬───────────────────────────────────────┘  ║
║                             ▼                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐  ║
║  │                     CONTEXT BUILDER                             │  ║
║  │                                                                  │  ║
║  │   Parent chunk fetch                                            │  ║
║  │     top-3 chunks → fetch parent section → richer context        │  ║
║  │                                                                  │  ║
║  │   Token budget enforcement                                      │  ║
║  │     memory + context + query must fit in 3800 tokens            │  ║
║  │                                                                  │  ║
║  │   Compress if over budget → SUMMARISER (Path B)                 │  ║
║  │                                                                  │  ║
║  │   Prompt assembly                                               │  ║
║  │     [system] + [memory] + [context] + [query]                   │  ║
║  └──────────────────────────┬───────────────────────────────────────┘  ║
║                             ▼                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐  ║
║  │                       SUMMARISER                                │  ║
║  │                                                                  │  ║
║  │   Path A — Conversation compression                             │  ║
║  │     Trigger : every 6 turns                                     │  ║
║  │     Input   : old turns from short-term buffer                  │  ║
║  │     Output  : rolling summary written to SQLite                 │  ║
║  │                                                                  │  ║
║  │   Path B — Context compression                                  │  ║
║  │     Trigger : retrieved context > 2000 tokens                   │  ║
║  │     Input   : top-3 chunks + parent context                     │  ║
║  │     Output  : compressed context → prompt builder               │  ║
║  │                                                                  │  ║
║  │   Model : qwen2:1.5b via Ollama (both paths)                    │  ║
║  └──────────────────────────────────────────────────────────────────┘  ║
╚════════════════════════════════════════╦════════════════════════════════╝
                                         ║
╔════════════════════════════════════════╩════════════════════════════════╗
║                     OLLAMA  localhost:11434                             ║
║                                                                         ║
║   qwen2.5:7b          4.7 GB    Primary generation · multi-hop         ║
║   qwen2:1.5b          1.0 GB    Fast tasks · summariser · classifier   ║
║                                                                         ║
╚═════════════════════════════════════════════════════════════════════════╝

Local embeddings run through Hugging Face sentence-transformers using
`BAAI/bge-small-en-v1.5` with 384-dim normalized vectors.

╔═════════════════════════════════════════════════════════════════════════╗
║               INGESTION PIPELINE  (chunker → caption → indexer)        ║
║                                                                         ║
║  chunker.py          → 1000-token chunks · 15% overlap · tables inline ║
║                                                                         ║
║  image_captioner.py                                                    ║
║    Input  : backend/extracted/images/*.png                             ║
║    Model  : Gemini 2.0 Flash  (API vision · index time only)           ║
║    Output : backend/extracted/image_captions.json                      ║
║    Action : captions appended to parent chunk text before embedding    ║
║                                                                         ║
║  indexer.py                                                             ║
║    FAISS build      BAAI/bge-small-en-v1.5 via sentence-transformers   ║
║      index type:    IndexHNSWFlat                                      ║
║      space:         cosine (inner product + L2 norm)                   ║
║      M:             32                                                  ║
║      efConstruction:200                                                 ║
║      efSearch:      100                                                 ║
║      dimensions:    384                                                 ║
║      persistent:    backend/db/faiss_store/index.faiss                 ║
║      id map:        backend/db/faiss_store/id_map.json                  ║
║      batch size:    50 chunks                                           ║
║                                                                         ║
║    BM25 build       rank_bm25 Okapi                                    ║
║      tokenized:     lower-cased word tokens                            ║
║      persistent:    backend/db/bm25_index.pkl                         ║
║                                                                         ║
║    Both indexes are idempotent — safe to re-run                        ║
╚═════════════════════════════════════════════════════════════════════════╝
```

---

## File Structure

Current layout (March 2026). Paths are relative to the **repository root** (`rag/`).

```
rag/
├── pyproject.toml
├── uv.lock
├── .gitignore
├── README.md
├── ARCHITECTURE.md
├── .env.example
│
├── frontend/
│   ├── app.py
│   └── README.md
│
└── backend/
    ├── settings.yaml
    ├── main.py
    ├── cli.py
    ├── embeddings.py
    │
    ├── api/
    │   ├── __init__.py
    │   ├── dependencies.py
    │   ├── query.py              ← POST /api/query
    │   └── schemas.py
    │
    ├── retriever/
    │   ├── retriever.py          ← FAISS + BM25 + RRF + dedup
    │   ├── context_builder.py
    │   └── rag_chain.py
    │
    ├── ingestion/
    │   ├── chunker.py
    │   ├── indexer.py
    │   ├── verify_index.py
    │   ├── image_captioner.py
    │   ├── acronym_extractor.py
    │   └── table_extractor.py
    │
    ├── extracted/                ← generated / versioned artifacts (examples)
    │   ├── documents.json
    │   ├── parent_index.json
    │   ├── acronyms.json
    │   ├── image_captions.json
    │   └── images/               ← figure crops (when present)
    │
    ├── data/
    │   └── raw/
    │       └── rag_structure_csv.csv   ← section map; place nasa_handbook.pdf here for ingestion
    │
    ├── db/
    │   ├── faiss_store/
    │   │   ├── index.faiss       ← created by indexer (may be gitignored)
    │   │   ├── id_map.json
    │   │   └── corpus.json       ← optional auxiliary dump
    │   └── bm25_index.pkl
    │
    ├── test/
    │   ├── test_rag.py
    │   ├── test_retriever.py
    │   ├── test_chain_live.py
    │   └── test_image_citations.py
    │
    ├── backup/
    │   ├── backup_chunker.py
    │   └── backup_chunk_2
    │
    └── Intstructions/            ← design / migration notes (folder name as in repo)
        ├── FAISS_MIGRATION_INSTRUCTIONS.md
        ├── HYBRID_EMBEDDING_INSTRUCTIONS.md
        ├── COPILOT_INSTRUCTIONS_OF_AI_IMPORVEMENT.md
        └── acronym.md
```

**Path resolution:** Python resolves `settings.yaml`, `data/`, `db/`, and `extracted/` relative to **`backend/`**. Values in `settings.yaml` such as `db/faiss_store/index.faiss` are joined against `backend/`. Load secrets from **`.env`** at the **repository root** (alongside `pyproject.toml`).

---

## settings.yaml

Single source of truth: **`backend/settings.yaml`**. The following is a representative excerpt; the file in the repo also includes `captioning` (Gemini), `gemini`, `groq`, and other keys.

```yaml
models:
  embedding: "BAAI/bge-small-en-v1.5"
  llm:
    provider: "groq"          # or "ollama"
    groq_profile: "llama3-fast"
    standard: "qwen2.5:7b"    # when provider is ollama

embeddings:
  provider: huggingface
  dimension: 384
  normalize: true
  query_prefix: "Represent this sentence for searching relevant passages: "
  providers:
    huggingface:
      model_name: "BAAI/bge-small-en-v1.5"
      device: "cpu"

ollama:
  base_url: "http://localhost:11434"
  timeout: 60

retrieval:
  vector_store: faiss
  faiss_index_path: db/faiss_store/index.faiss      # relative to backend/
  faiss_id_map_path: db/faiss_store/id_map.json
  faiss_checkpoint_every: 10
  hnsw_M: 32
  hnsw_ef_construction: 200
  hnsw_ef_search: 100
  dense_top_k: 10
  sparse_top_k: 10
  rerank_top_n: 3
  rrf_k: 60
  confidence_threshold: 0.6
  dedup_threshold: 0.92
```

---

## Noise removal

Repeated header/footer phrases and boilerplate are filtered during chunking. In this codebase, noise patterns are defined in **`backend/ingestion/chunker.py`** (for example the `NOISE_VARIABLES` list), not in a separate JSON config file. Adjust there and re-run the ingestion scripts if you need stricter or looser filtering.

---

## include_in_rag Column Reference

| Value | Applies To | Action |
|---|---|---|
| `yes` | Chapters 1–6 · Appendices B C G I J T | Full text + table + image extraction · chunked · captioned · indexed |
| `acronym` | Appendix A | Acronym dictionary extraction only · not chunked |
| `no` | TOC · Preface · Ack · Appendices D E F H K L M N P R S · References · Bibliography | Skipped entirely · never touched |

---

## Model Downloads

```bash
ollama pull qwen2.5:7b        # 4.7 GB — primary generation
ollama pull qwen2:1.5b        # 1.0 GB — fast tasks
ollama pull qwen2.5vl:7b      # 6.0 GB — image captioning
```

The embedding model `BAAI/bge-small-en-v1.5` downloads automatically from Hugging Face on first use through `sentence-transformers`. A separate cross-encoder re-ranker is **not** bundled in this repository (see Implementation Status).

---

## Run Commands

From the repository root (uses [uv](https://github.com/astral-sh/uv)):

```bash
uv sync                                              # install dependencies
uv run uvicorn backend.main:app --reload             # API
uv run streamlit run frontend/app.py                 # Streamlit UI
uv run python -m backend.cli "Your question"       # optional CLI
uv run pytest                                        # tests under backend/test/
```

Ingestion is run via the scripts in `backend/ingestion/` (for example `chunker.py`, `image_captioner.py`, `indexer.py`) rather than a `Makefile` in this repo.

---

## What Changed from Original Architecture

| Component | Before | After | Reason |
|---|---|---|---|
| Vector store | ChromaDB (hnswlib wrapped) | FAISS IndexHNSWFlat | CPU SIMD perf · direct tuning · no server overhead |
| HNSW params | M=16 default · no ef tuning | M=32 · efConstruction=200 · efSearch=100 | Better recall on ~3K chunk corpus |
| Metadata store | ChromaDB built-in | Parallel `id_map.json` | FAISS stores vectors only |
| Image handling | Not in retrieval pipeline | Gemini vision captions appended to chunks (index time) | Images searchable via text queries |
| LLM / vision APIs | — | Optional Groq (chat), Gemini (captioning) via env keys | Configured in `backend/settings.yaml` |
| Post-RRF dedup | Not present | cosine sim > threshold (`dedup_threshold`) | Implemented in `backend/retriever/retriever.py` |
| New file | — | `backend/ingestion/image_captioner.py` | Captioning step before indexing |
| New file | — | `backend/extracted/image_captions.json` | Persisted captions · avoid re-calling API |

*Architecture updated post Hireathon 2026 design session.*

Embeddings for retrieval run locally (Hugging Face). **Query-time** generation may use **Groq** or **Ollama** per settings; **index-time** figure captioning uses the **Gemini** API when configured.

---

## Implementation Status

**✅ Currently implemented in this repo:**
- **HybridRetriever** (`backend/retriever/retriever.py`) — FAISS dense search + BM25 sparse search, **RRF fusion** (`rrf_k` from settings), **dedup** (`dedup_threshold`), query embeddings via **`backend/embeddings.py`** (Hugging Face / Ollama)
- **FAISS + BM25 index build** — `backend/ingestion/indexer.py`
- **Gemini** image captioning pipeline — `backend/ingestion/image_captioner.py` (index time); captions cached under `backend/extracted/`
- **Context assembly + LLM chain** — `context_builder.py`, `rag_chain.py` (Groq or Ollama per `settings.yaml`)
- **FastAPI** — `backend/main.py`, routes under `backend/api/`

**⚠️ Aspirational / not wired as in the diagram above:**
- **Cross-encoder re-ranker** — listed in the technology table and diagram; not implemented in `backend/retriever/` (no separate cross-encoder model; `rerank_top_n` exists in `settings.yaml` but is not referenced by Python code in this snapshot)
- **Memory manager, SQLite sessions, eval dashboard** — shown in the large architecture figure; the current tree is API + Streamlit + RAG chain only
