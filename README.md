# 🧬 MedRAG Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-v1.2-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.38-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <b>A production-grade Retrieval-Augmented Generation (RAG) agent for biomedical question answering.</b><br/>
  Ingests real data from PubMed and OpenFDA · Indexes into FAISS · Serves evidence-based cited answers<br/>
  via a LangChain ReAct agent, REST API, and interactive Streamlit UI.
</p>

---

## 📸 Demo

> **Query:** *"What are the latest treatments for Type 2 Diabetes and what are metformin's interactions?"*

```
🔍 Agent reasoning...
  → [search_pubmed_literature]     "type 2 diabetes treatment 2024"
  → [lookup_fda_drug_info]         "metformin"
  → [search_pubmed_literature]     "GLP-1 receptor agonists cardiovascular outcomes"
  → [get_adverse_event_statistics] "semaglutide"

📋 Answer:
Current evidence supports a tiered approach to T2D management. First-line therapy
remains metformin, with established efficacy for HbA1c reduction and a favorable
safety profile over 60+ years of use. GLP-1 receptor agonists (semaglutide,
liraglutide) have demonstrated significant cardiovascular benefits beyond glycemic
control in recent landmark trials...

📚 Sources:
  • Smith J et al. (2024) — PMID: 38234567 — "Comparative GLP-1 efficacy..."
  • FDA Drug Label — Metformin HCl (Glucophage) — NDA 020357
  • Johnson A et al. (2023) — PMID: 37891234 — "Cardiovascular outcomes SUSTAIN..."
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MedRAG Pipeline                              │
│                                                                      │
│   DATA SOURCES            ETL PIPELINE            AGENT LAYER       │
│  ┌───────────┐          ┌─────────────┐         ┌──────────────┐   │
│  │  PubMed   │ ───────► │  Extract    │         │  LangChain   │   │
│  │  E-utils  │          │  (clients)  │         │  ReAct Agent │   │
│  └───────────┘          └──────┬──────┘         └──────┬───────┘   │
│  ┌───────────┐                 │ Transform              │           │
│  │  OpenFDA  │ ───────────────►│  Chunk + Embed │ ┌────▼───────┐   │
│  │  API      │          ┌──────▼──────┐         │ │   Tools    │   │
│  └───────────┘          │    FAISS    │◄────────►│ │ • PubMed  │   │
│                         │  Vector DB  │  Load    │ │ • FDA     │   │
│                         └─────────────┘         │ │ • FAERS   │   │
│                                                  └────────────┘    │
│   INTERFACES                                                        │
│  ┌────────────┐   ┌───────────┐                                     │
│  │ Streamlit  │   │  FastAPI  │ ──────────────► AgentExecutor       │
│  │     UI     │   │  REST API │                                     │
│  └────────────┘   └───────────┘                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Data flow:** APIs → `Article`/`DrugRecord` dataclasses → chunked `Document` objects → FAISS vectors → LangChain retriever → ReAct agent tools → cited answer.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔄 **End-to-end ETL** | Ingests real PubMed abstracts and FDA drug labels via public APIs |
| 🧩 **Smart chunking** | Configurable chunk size/overlap, separate strategies per document type |
| 🤖 **ReAct Agent** | LangChain agent autonomously chains tools based on question type |
| 🔍 **Multi-source RAG** | Queries PubMed literature AND live FDA data in a single answer |
| 📋 **Source citations** | Every answer references PMID numbers and FDA NDA identifiers |
| 🎛️ **Streamlit UI** | Interactive dark-themed interface with query history and pipeline control |
| ⚡ **FastAPI** | Async REST endpoints with auto-generated OpenAPI docs |
| 🐳 **Docker** | Single `docker-compose up` for full stack deployment |
| ✅ **PyTest suite** | Mocked unit tests — no real API calls needed in CI |
| 🔁 **Retry logic** | `tenacity`-powered exponential backoff for all external API calls |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- PubMed and OpenFDA are **free** — no key required to start

### 1. Clone & install

```bash
git clone https://github.com/giulio-quaglia/medrag-pipeline.git
cd medrag-pipeline

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Add your OPENAI_API_KEY — everything else is optional
```

### 3. Build the vector index (ETL pipeline)

```bash
python scripts/run_pipeline.py \
  --query "type 2 diabetes treatment" \
  --max_results 50 \
  --drugs metformin semaglutide liraglutide
```

This will:
1. Search PubMed and download article abstracts
2. Fetch FDA drug labels for the listed drugs
3. Chunk, embed (OpenAI), and save a FAISS index to `data/processed/faiss_index/`

### 4. Query the agent

**Streamlit UI (recommended)**
```bash
streamlit run app.py
# Open http://localhost:8501
```

**CLI**
```bash
python scripts/query_agent.py "What are the side effects of metformin?"
python scripts/query_agent.py --interactive   # REPL session
```

**REST API**
```bash
uvicorn src.api.main:app --reload
# Docs at http://localhost:8000/docs
```

**Docker**
```bash
docker-compose up --build
```

---

## 📁 Project Structure

```
medrag-pipeline/
│
├── app.py                          # Streamlit UI (dark clinical theme)
│
├── src/
│   ├── ingestion/
│   │   ├── pubmed_client.py        # PubMed E-utilities API → Article dataclass
│   │   └── openfda_client.py       # OpenFDA API → DrugRecord dataclass
│   │
│   ├── pipeline/
│   │   └── vector_store.py         # Chunking, embedding, FAISS index management
│   │
│   ├── agent/
│   │   ├── tools.py                # 4 LangChain @tool functions
│   │   ├── agent.py                # MedRAGAgent + AgentResponse
│   │   └── prompts.py              # System prompt + ReAct template
│   │
│   └── api/
│       └── main.py                 # FastAPI — /query, /health, /pipeline/run
│
├── scripts/
│   ├── run_pipeline.py             # ETL CLI with rich progress output
│   └── query_agent.py              # Query CLI + interactive REPL
│
├── tests/
│   └── test_pipeline.py            # PyTest suite (fully mocked)
│
├── data/
│   ├── raw/                        # gitignored
│   └── processed/faiss_index/      # gitignored
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| LLM Framework | LangChain v1.2 + LangGraph | Industry-standard agent orchestration |
| LLM | OpenAI GPT-4.1-mini | Best quality/cost ratio for RAG tasks |
| Embeddings | `text-embedding-3-small` | 1536-dim, fast, cost-efficient |
| Vector Store | FAISS (CPU) | Local, no infra needed, production-ready |
| Data Sources | PubMed E-utilities + OpenFDA | Free, authoritative, real biomedical data |
| ETL | Python + Biopython + xmltodict | Robust XML parsing for PubMed responses |
| API | FastAPI + Uvicorn | Async, typed, auto-generated OpenAPI docs |
| UI | Streamlit | Rapid, data-friendly interface |
| Testing | PyTest + unittest.mock | CI-safe, no external API calls |
| Infra | Docker + Docker Compose | Reproducible one-command deployment |
| Reliability | Tenacity | Exponential backoff on all API calls |

---

## 🤖 Agent Tools

The ReAct agent selects tools autonomously based on the question. Tool descriptions are precise — they are what the LLM reads to decide which tool to call:

| Tool | Data Source | Use Case |
|------|-------------|----------|
| `search_pubmed_literature` | FAISS index | Mechanisms, clinical evidence, epidemiology |
| `lookup_fda_drug_info` | OpenFDA API (live) | Official indications, warnings, interactions |
| `search_drug_in_literature` | FAISS index | Clinical trials for a specific drug |
| `get_adverse_event_statistics` | OpenFDA FAERS (live) | Real-world post-market safety signals |

**Example reasoning trace:**

```
Question: What are the cardiovascular risks of semaglutide?

Thought: I need FDA warnings AND clinical outcome data.
Action: lookup_fda_drug_info
Action Input: semaglutide
Observation: FDA Label — Ozempic... thyroid C-cell tumors, pancreatitis risk...

Thought: Now check RCT evidence for CV outcomes.
Action: search_pubmed_literature
Action Input: semaglutide cardiovascular outcomes SUSTAIN SOUL trial
Observation: PMID:38234567 — "SOUL trial: CV death reduction 26%..."

Thought: I have comprehensive data from both sources.
Final Answer: Semaglutide carries FDA warnings for thyroid tumors and
pancreatitis. However, the SOUL trial (2024, PMID:38234567) demonstrated
a 26% relative reduction in cardiovascular death compared to placebo...
```

---

## ⚙️ Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | LLM + embedding calls |
| `PUBMED_EMAIL` | Recommended | Increases PubMed rate limit identification |
| `PUBMED_API_KEY` | Optional | 10 req/s (vs 3 req/s without key) |
| `OPENFDA_API_KEY` | Optional | Higher OpenFDA rate limits |
| `VECTOR_STORE_PATH` | Optional | Default: `./data/processed/faiss_index` |
| `LANGCHAIN_TRACING_V2` | Optional | Enable LangSmith observability |
| `LANGCHAIN_API_KEY` | Optional | LangSmith key (if tracing enabled) |

---

## 🧪 Tests

```bash
pytest tests/ -v                              # All tests
pytest tests/ --cov=src --cov-report=term    # With coverage
pytest tests/test_pipeline.py::TestArticle   # Specific class
```

All tests use mocks — no real API calls, CI-safe.

---

## 📊 CLI Reference

```bash
# ETL pipeline
python scripts/run_pipeline.py \
  --query "hypertension ACE inhibitors" \
  --max_results 40 \
  --drugs lisinopril amlodipine \
  --append                          # Add to existing index

# Query agent
python scripts/query_agent.py "question here"  # Single question
python scripts/query_agent.py --verbose         # Show reasoning steps
python scripts/query_agent.py --interactive     # REPL session
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Agent status + vector store stats |
| `POST` | `/query` | Submit a biomedical question |
| `POST` | `/pipeline/run` | Trigger ETL pipeline (background task) |

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the contraindications of warfarin?"}'
```

---

## 🗺️ Roadmap

- [ ] Streaming responses in Streamlit (token-by-token)
- [ ] Full-text ingestion from PubMed Central (PMC) open access papers
- [ ] LangGraph refactor for explicit state graph + better observability
- [ ] Persistent cross-session memory (SQLite checkpointer)
- [ ] Retrieval re-ranking (Cohere / BGE reranker)
- [ ] RAGAS evaluation suite (faithfulness, answer relevancy, context recall)
- [ ] Azure Container Apps deployment template

---

## ⚕️ Disclaimer

This tool is intended for **research and educational purposes only**. It synthesizes publicly available scientific literature and FDA drug label data. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical decisions.

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

## 👤 Author

**Giulio Quaglia** — AI & Data Architect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-giulio--quaglia-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/giulio-quaglia)
[![GitHub](https://img.shields.io/badge/GitHub-gqkd-181717?style=flat&logo=github)](https://github.com/gqkd)
