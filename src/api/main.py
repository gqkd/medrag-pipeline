"""
src/api/main.py
────────────────
FastAPI application exposing the MedRAG agent as a REST API.

Endpoints:
  GET  /health            — liveness + agent status
  POST /query             — submit a biomedical question
  POST /pipeline/run      — trigger the ETL pipeline (background task)
  GET  /docs              — auto-generated OpenAPI docs (Swagger UI)
  GET  /redoc             — ReDoc documentation

Run locally::

    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    QueryRequest,
    QueryResponse,
    PipelineRequest,
    PipelineResponse,
    HealthResponse,
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Application state
# ─────────────────────────────────────────────────────────────────────────────

_agent = None  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load agent on startup; release on shutdown."""
    global _agent
    log.info("Starting MedRAG API — loading agent...")
    try:
        from src.agent.agent import build_agent
        _agent = build_agent(verbose=False)
        log.info("Agent loaded — %d vectors indexed", _agent.vector_store.total_vectors)
    except RuntimeError as exc:
        log.warning("Agent not available at startup: %s", exc)
        log.warning("Run the ETL pipeline then restart the server.")
    except Exception as exc:
        log.error("Unexpected error loading agent: %s", exc, exc_info=True)
    yield
    _agent = None
    log.info("MedRAG API shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedRAG Pipeline API",
    description=(
        "**Biomedical RAG agent** powered by PubMed literature and FDA drug data.\n\n"
        "Ask any medical or pharmacological question and receive evidence-based "
        "answers with traceable source citations.\n\n"
        "**Quick start:**\n"
        "1. Build the vector index: `python scripts/run_pipeline.py --query 'your topic'`\n"
        "2. POST to `/query` with your question\n\n"
        "> ⚕️ For research use only — not a substitute for medical advice."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Dependency helpers
# ─────────────────────────────────────────────────────────────────────────────


def _require_agent():
    """Raise 503 if the agent hasn't been loaded yet."""
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Agent not available. "
                "Build the vector index first:\n"
                "  python scripts/run_pipeline.py --query 'your topic' --max_results 50"
            ),
        )
    return _agent


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
async def health() -> HealthResponse:
    """
    Returns the current status of the API and agent.
    Use this endpoint to verify the vector index is loaded before querying.
    """
    stats = _agent.vector_store.get_stats() if _agent else {"status": "not_loaded"}
    return HealthResponse(
        status="ok",
        agent_loaded=_agent is not None,
        vector_store_stats=stats,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["Agent"],
    summary="Ask a biomedical question",
)
async def query_agent(request: QueryRequest) -> QueryResponse:
    """
    Submit a natural-language biomedical question to the MedRAG agent.

    The agent will:
    1. Analyse the question and decide which tools to invoke
    2. Search PubMed literature and/or FDA drug data
    3. Return a synthesised, cited answer

    **Example questions:**
    - *"What are the latest treatments for Type 2 Diabetes?"*
    - *"What are the drug interactions of warfarin?"*
    - *"Compare GLP-1 agonists and SGLT-2 inhibitors for cardiovascular risk."*
    """
    agent = _require_agent()

    t0 = time.perf_counter()
    try:
        response = agent.query(request.question)
    except Exception as exc:
        log.error("Query failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent error: {exc}",
        )
    elapsed = time.perf_counter() - t0

    return QueryResponse(
        question=request.question,
        answer=response.answer,
        sources=response.sources,
        tools_used=sorted(set(response.tools_used)),
        processing_time_seconds=round(elapsed, 2),
        intermediate_steps=(
            str(response.intermediate_steps) if request.verbose else None
        ),
    )


@app.post(
    "/pipeline/run",
    response_model=PipelineResponse,
    tags=["Pipeline"],
    summary="Trigger ETL pipeline",
    status_code=status.HTTP_202_ACCEPTED,
)
async def run_pipeline(
    request: PipelineRequest,
    background_tasks: BackgroundTasks,
) -> PipelineResponse:
    """
    Trigger the ETL pipeline to ingest new data into the vector index.
    Runs as a background task — returns immediately with 202 Accepted.
    Poll `/health` to check when the new data is available.
    """

    def _run_etl() -> None:
        global _agent
        try:
            from scripts.run_pipeline import run_pipeline as etl
            etl(
                query=request.query,
                max_results=request.max_results,
                drug_names=request.drug_names,
                append=request.append,
            )
            # Reload the agent with the updated index
            from src.agent.agent import build_agent
            _agent = build_agent(verbose=False)
            log.info("Agent reloaded after ETL pipeline completion.")
        except Exception as exc:
            log.error("Background ETL pipeline failed: %s", exc, exc_info=True)

    background_tasks.add_task(_run_etl)

    return PipelineResponse(
        status="accepted",
        message=(
            f"ETL pipeline started for query: '{request.query}'. "
            f"Poll /health to monitor progress."
        ),
        query=request.query,
    )
