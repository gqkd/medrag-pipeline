# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing   # with coverage
pytest tests/test_pipeline.py::TestArticle           # single class

# Lint / format
ruff check src/ tests/ scripts/ app.py
black src/ tests/ scripts/ app.py

# Run the ETL pipeline (must run before querying)
python src/scripts/run_pipeline.py \
  --query "type 2 diabetes treatment" \
  --max_results 50 \
  --drugs metformin semaglutide

# Interfaces
streamlit run app.py                          # Streamlit UI at :8501
uvicorn src.api.main:app --reload             # FastAPI at :8000/docs
python src/scripts/query_agent.py "question"  # CLI single query
python src/scripts/query_agent.py --interactive  # CLI REPL

# Docker
docker-compose up --build

# Makefile shortcuts
make test | make lint | make format | make run-ui | make run-api
```

## Required setup

Copy `env.example` to `.env` and set at minimum:
- `OPENAI_API_KEY` — used for both LLM (GPT-4.1-mini) and embeddings (text-embedding-3-small)

The FAISS index (`data/processed/faiss_index/`) must be built by running `run_pipeline.py` before the agent can answer questions. All three interfaces (`app.py`, API, CLI) call `build_agent()`, which calls `vs.load()` and raises `RuntimeError` if the index is missing.

## Architecture

**Data flow:** External APIs → typed dataclasses → LangChain `Document` chunks → FAISS vectors → ReAct agent tools → cited answer.

### Layer breakdown

| Layer | Files | Role |
|---|---|---|
| Config | [src/config.py](src/config.py) | Pydantic `Settings` singleton; all tunable values read from `.env` |
| Ingestion | [src/ingestion/pubmed_client.py](src/ingestion/pubmed_client.py), [openfda_client.py](src/ingestion/openfda_client.py) | Fetch from PubMed (Biopython/XML) and OpenFDA REST; return `Article` / `DrugRecord` dataclasses |
| ETL | [src/pipeline/vector_store.py](src/pipeline/vector_store.py) | Chunk → embed → FAISS; two splitters: 512-token for abstracts, 1024-token for FDA labels |
| Agent tools | [src/agent/tools.py](src/agent/tools.py) | Four `@tool` functions wired with injected `MedRAGVectorStore` + `OpenFDAClient`; tool descriptions are the LLM's decision contract — keep them precise |
| Agent | [src/agent/agent.py](src/agent/agent.py) | `MedRAGAgent` wraps `AgentExecutor`; `build_agent()` is the single entry point for all consumers |
| Prompts | [src/agent/prompts.py](src/agent/prompts.py) | System prompt + ReAct template |
| API | [src/api/main.py](src/api/main.py) | FastAPI: `POST /query`, `GET /health`, `POST /pipeline/run` (background task) |
| UI | [app.py](app.py) | Streamlit dark-themed UI with query history and sidebar ETL panel |

### Key design decisions

- **`build_agent()`** in `agent.py` is the canonical factory used by all three interfaces. It loads the vector store from disk, then constructs `MedRAGAgent` with injected dependencies.
- **Tools use dependency injection** via `build_tools(vector_store, fda_client)` — no module-level singletons in tools.
- **`lookup_fda_drug_info`** makes live OpenFDA API calls (not from FAISS); the other three tools query the FAISS index. `get_adverse_event_statistics` calls the FAERS endpoint.
- **FAISS index** is flat (`IndexFlatL2`); suitable for ~200k vectors. For larger corpora, switch to `IndexIVFFlat`.
- **All settings** flow through the `settings` singleton (`from src.config import settings`); override via `.env` or environment variables.

### Tests

All tests in [src/tests/test_pipeline.py](src/tests/test_pipeline.py) mock OpenAI, FAISS, PubMed, and FDA — no network access or `.env` required. Fixtures are in [src/tests/conftest.py](src/tests/conftest.py). Tests are organized by class: `TestArticle`, `TestDrugRecord`, `TestAdverseEventSummary`, `TestPubMedClientParsing`, `TestMedRAGVectorStore`, `TestOpenFDAClientParsing`, `TestToolBuilding`.
