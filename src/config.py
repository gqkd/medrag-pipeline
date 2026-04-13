"""
src/config.py
──────────────
Centralized configuration via Pydantic Settings.
All values are read from environment variables or .env file.
Import `settings` anywhere in the project.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────
    openai_api_key: str = ""
    default_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = 0.1
    max_agent_iterations: int = 8

    # ── PubMed ────────────────────────────────────────────────────
    pubmed_email: str = "medrag@example.com"
    pubmed_api_key: str = ""
    pubmed_batch_size: int = 20

    # ── OpenFDA ───────────────────────────────────────────────────
    openfda_api_key: str = ""

    # ── Vector Store ──────────────────────────────────────────────
    vector_store_path: Path = Path("./data/processed/faiss_index")
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_k: int = 5

    # ── API ───────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── LangSmith (optional) ──────────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "medrag-pipeline"

    # ── Logging ───────────────────────────────────────────────────
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


# Module-level singleton for convenience: `from src.config import settings`
settings = get_settings()
