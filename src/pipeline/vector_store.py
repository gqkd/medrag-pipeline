"""
src/pipeline/vector_store.py
──────────────────────────────
ETL — Transform + Load layer.

Responsibilities:
  1. Chunk documents (articles, drug records) into overlapping text segments
  2. Embed each segment via OpenAI text-embedding-3-small
  3. Store + persist a FAISS vector index to disk
  4. Expose a retriever interface consumed by the LangChain agent tools

Chunking strategy:
  • PubMed abstracts  → 512 tokens / 64 overlap  (short → precise recall)
  • FDA drug labels   → 1024 tokens / 128 overlap (long → context continuity)

The FAISS index is flat (IndexFlatL2) by default — suitable for up to
~200k vectors. For larger corpora switch to IndexIVFFlat or use pgvector.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.pubmed_client import Article
from src.ingestion.openfda_client import DrugRecord

log = logging.getLogger(__name__)


class MedRAGVectorStore:
    """
    Thin wrapper around a FAISS vector store with domain-specific chunking.

    Example usage::

        vs = MedRAGVectorStore()
        vs.add_articles(articles)
        vs.add_drug_records(records)
        vs.save()

        # Later:
        vs = MedRAGVectorStore()
        vs.load()
        retriever = vs.as_retriever(k=5)
    """

    def __init__(
        self,
        index_path: Path | None = None,
        embedding_model: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        from src.config import settings

        self.index_path: Path = index_path or settings.vector_store_path
        self._embeddings = OpenAIEmbeddings(
            model=embedding_model or settings.embedding_model
        )

        _chunk_size = chunk_size or settings.chunk_size
        _chunk_overlap = chunk_overlap or settings.chunk_overlap

        # Splitter for short biomedical abstracts
        self._abstract_splitter = RecursiveCharacterTextSplitter(
            chunk_size=_chunk_size,
            chunk_overlap=_chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            length_function=len,
        )

        # Splitter for longer FDA label sections
        self._label_splitter = RecursiveCharacterTextSplitter(
            chunk_size=_chunk_size * 2,
            chunk_overlap=_chunk_overlap * 2,
            separators=["\n\n", "\n", ". ", " "],
            length_function=len,
        )

        self._store: FAISS | None = None

    # ─────────────────────────────────────────────────────────────
    # Ingestion
    # ─────────────────────────────────────────────────────────────

    def add_articles(self, articles: list[Article]) -> int:
        """
        Chunk and embed a list of PubMed articles.
        Returns the number of document chunks added to the index.
        """
        if not articles:
            log.warning("add_articles called with empty list")
            return 0

        documents: list[Document] = []
        for article in articles:
            chunks = self._abstract_splitter.split_text(article.full_text)
            for idx, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        **article.to_metadata(),
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "doc_type": "pubmed_article",
                    },
                ))

        log.info(
            "Embedding %d chunks from %d PubMed articles",
            len(documents),
            len(articles),
        )
        return self._upsert(documents)

    def add_drug_records(self, records: list[DrugRecord]) -> int:
        """
        Chunk and embed a list of FDA drug records.
        Returns the number of document chunks added to the index.
        """
        if not records:
            return 0

        documents: list[Document] = []
        for record in records:
            chunks = self._label_splitter.split_text(record.full_text)
            for idx, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        **record.to_metadata(),
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "doc_type": "fda_drug_label",
                    },
                ))

        log.info(
            "Embedding %d chunks from %d FDA drug records",
            len(documents),
            len(records),
        )
        return self._upsert(documents)

    def _upsert(self, documents: list[Document]) -> int:
        """Internal: embed documents and merge into the FAISS index."""
        if not documents:
            return 0
        if self._store is None:
            self._store = FAISS.from_documents(documents, self._embeddings)
        else:
            self._store.add_documents(documents)
        return len(documents)

    # ─────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> None:
        """Serialize the FAISS index and docstore to disk."""
        if self._store is None:
            raise RuntimeError("Nothing to save — add documents first.")
        dest = path or self.index_path
        dest.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(dest))
        log.info("Vector store saved → %s  (%d vectors)", dest, self.total_vectors)

    def load(self, path: Path | None = None) -> bool:
        """
        Load a previously saved FAISS index from disk.
        Returns ``True`` on success, ``False`` if the index does not exist yet.
        """
        src = path or self.index_path
        index_file = src / "index.faiss"
        if not index_file.exists():
            log.info("No vector index at %s — run the ETL pipeline first.", src)
            return False
        self._store = FAISS.load_local(
            str(src),
            self._embeddings,
            allow_dangerous_deserialization=True,
        )
        log.info("Vector store loaded ← %s  (%d vectors)", src, self.total_vectors)
        return True

    # ─────────────────────────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────────────────────────

    def as_retriever(
        self,
        k: int | None = None,
        source_filter: str | None = None,
        score_threshold: float | None = None,
    ) -> VectorStoreRetriever:
        """
        Return a LangChain :class:`VectorStoreRetriever` for use in agent tools.

        Args:
            k: Number of documents to return per query (default from settings).
            source_filter: Optional ``"pubmed"`` or ``"openfda"`` to restrict
                           retrieval to a single data source.
            score_threshold: If set, use similarity score threshold instead
                             of fixed-k retrieval.
        """
        self._require_store()
        from src.config import settings

        search_kwargs: dict[str, Any] = {"k": k or settings.retrieval_k}
        if source_filter:
            search_kwargs["filter"] = {"source": source_filter}

        if score_threshold is not None:
            return self._store.as_retriever(  # type: ignore[union-attr]
                search_type="similarity_score_threshold",
                search_kwargs={**search_kwargs, "score_threshold": score_threshold},
            )
        return self._store.as_retriever(search_kwargs=search_kwargs)  # type: ignore[union-attr]

    def similarity_search(
        self, query: str, k: int = 5, source_filter: str | None = None
    ) -> list[Document]:
        """
        Direct similarity search — used internally by agent tools.

        Args:
            query: Natural-language query string.
            k: Number of results.
            source_filter: Optionally restrict to ``"pubmed"`` or ``"openfda"``.
        """
        self._require_store()
        kwargs: dict[str, Any] = {}
        if source_filter:
            kwargs["filter"] = {"source": source_filter}
        return self._store.similarity_search(query, k=k, **kwargs)  # type: ignore[union-attr]

    # ─────────────────────────────────────────────────────────────
    # Introspection
    # ─────────────────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        """Number of vectors currently in the index."""
        if self._store is None:
            return 0
        return self._store.index.ntotal

    def get_stats(self) -> dict[str, Any]:
        """Return a status dict suitable for the health endpoint."""
        if self._store is None:
            return {"status": "empty", "total_vectors": 0, "index_path": str(self.index_path)}
        return {
            "status": "loaded",
            "total_vectors": self.total_vectors,
            "index_path": str(self.index_path),
        }

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    def _require_store(self) -> None:
        if self._store is None:
            raise RuntimeError(
                "Vector store is not initialized.\n"
                "Either call .load() to load an existing index, or\n"
                "call .add_articles() / .add_drug_records() to build one."
            )
