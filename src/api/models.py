"""
src/api/models.py
──────────────────
Pydantic models for FastAPI request validation and response serialization.
Keeping models separate from main.py keeps the API layer clean and testable.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="The biomedical question to answer.",
        examples=["What are the side effects of metformin?"],
    )
    verbose: bool = Field(
        default=False,
        description="If true, include the agent's intermediate reasoning steps.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Override the default LLM. E.g. 'gpt-4.1' for harder questions.",
        examples=["gpt-4.1-mini", "gpt-4.1"],
    )

    model_config = {"json_schema_extra": {
        "example": {
            "question": "What does the literature say about GLP-1 agonists for weight loss?",
            "verbose": False,
        }
    }}


class PipelineRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        description="PubMed search query. Supports full PubMed syntax.",
        examples=["type 2 diabetes GLP-1 treatment 2024"],
    )
    max_results: int = Field(
        default=30,
        ge=1,
        le=200,
        description="Maximum number of PubMed articles to ingest.",
    )
    drug_names: list[str] = Field(
        default_factory=list,
        description="Drug names to fetch FDA label data for.",
        examples=[["metformin", "semaglutide"]],
    )
    append: bool = Field(
        default=False,
        description="Append to the existing FAISS index instead of rebuilding it.",
    )


# ── Responses ─────────────────────────────────────────────────────────────────

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    processing_time_seconds: float
    intermediate_steps: Optional[str] = Field(
        default=None,
        description="Agent reasoning trace (only present when verbose=true).",
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    agent_loaded: bool
    vector_store_stats: dict


class PipelineResponse(BaseModel):
    status: str
    message: str
    query: str
    estimated_articles: Optional[int] = None
