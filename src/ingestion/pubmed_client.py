"""
src/ingestion/pubmed_client.py
──────────────────────────────
ETL — Extract layer: PubMed E-utilities API client.

Fetches article abstracts from NCBI PubMed. The free tier allows 3 req/s;
with a free NCBI API key (https://www.ncbi.nlm.nih.gov/account/) it rises
to 10 req/s. No account is required to get started.

Public API docs: https://www.ncbi.nlm.nih.gov/books/NBK25499/
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Iterator

import requests
import xmltodict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

log = logging.getLogger(__name__)

_PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_RETRY_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.HTTPError,
)


# ─────────────────────────────────────────────────────────────────────────────
# Domain model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Article:
    """
    A single PubMed article — the atomic unit flowing through the pipeline.
    All fields are plain Python types; no external dependencies.
    """

    pmid: str
    title: str
    abstract: str
    authors: list[str]
    journal: str
    pub_date: str
    keywords: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)
    doi: str = ""
    source: str = "pubmed"

    # ── Computed properties used downstream ──────────────────────

    @property
    def full_text(self) -> str:
        """Title + abstract concatenated — the text that gets embedded."""
        parts = [f"Title: {self.title}", f"\nAbstract: {self.abstract}"]
        if self.keywords:
            parts.append(f"\nKeywords: {', '.join(self.keywords[:8])}")
        return "".join(parts)

    @property
    def citation(self) -> str:
        """Short human-readable citation string."""
        authors = self.authors[:3]
        author_str = ", ".join(authors)
        if len(self.authors) > 3:
            author_str += " et al."
        year = self.pub_date if self.pub_date != "Unknown" else "n.d."
        return f"{author_str} ({year}) — PMID: {self.pmid} — {self.title[:80]}"

    def to_metadata(self) -> dict[str, str]:
        """Flat dict written into FAISS document metadata."""
        return {
            "pmid": self.pmid,
            "title": self.title[:200],
            "authors": "; ".join(self.authors[:5]),
            "journal": self.journal,
            "pub_date": self.pub_date,
            "doi": self.doi,
            "keywords": "; ".join(self.keywords[:10]),
            "mesh_terms": "; ".join(self.mesh_terms[:10]),
            "citation": self.citation,
            "source": self.source,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────


class PubMedClient:
    """
    Thin wrapper around the NCBI E-utilities REST API.

    Usage::

        client = PubMedClient()
        articles = client.search_and_fetch("metformin diabetes", max_results=30)
        for article in articles:
            print(article.citation)
    """

    def __init__(
        self,
        email: str | None = None,
        api_key: str | None = None,
        batch_size: int = 20,
        request_timeout: int = 30,
    ) -> None:
        from src.config import settings

        self.email = email or settings.pubmed_email
        self.api_key = api_key or settings.pubmed_api_key or ""
        self.batch_size = batch_size
        self.timeout = request_timeout

        # Base params injected into every request
        self._base: dict[str, str] = {"email": self.email, "tool": "medrag-pipeline"}
        if self.api_key:
            self._base["api_key"] = self.api_key

        # Rate-limit: 10 req/s with key, 3 without (use 0.12 s / 0.38 s delay)
        self._delay = 0.12 if self.api_key else 0.38

        self._session = requests.Session()

    # ── Private helpers ──────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        reraise=True,
    )
    def _get(self, endpoint: str, params: dict) -> dict:
        url = f"{_PUBMED_BASE}/{endpoint}"
        merged = {**self._base, **params}
        resp = self._session.get(url, params=merged, timeout=self.timeout)
        resp.raise_for_status()
        return xmltodict.parse(resp.text)

    # ── Public API ───────────────────────────────────────────────

    def search(self, query: str, max_results: int = 20) -> list[str]:
        """
        Run a PubMed query and return a list of PMIDs (strings).

        Supports full PubMed syntax including boolean operators, field tags,
        MeSH qualifiers, and date filters.

        Examples::

            client.search("metformin[MeSH] AND type 2 diabetes", max_results=50)
            client.search("COVID-19 long term outcomes 2023[pdat]")
        """
        log.info("PubMed search | query=%r | max=%d", query, max_results)
        data = self._get("esearch.fcgi", {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "xml",
            "sort": "relevance",
            "usehistory": "n",
        })
        result = data.get("eSearchResult", {})
        id_list = result.get("IdList", {}).get("Id", [])
        if isinstance(id_list, str):
            id_list = [id_list]
        log.info("PubMed search → %d PMIDs", len(id_list))
        return id_list

    def fetch_articles(self, pmids: list[str]) -> Iterator[Article]:
        """
        Stream full article records for the given PMIDs.

        Yields :class:`Article` objects one at a time. Articles without
        an abstract are silently skipped (not useful for RAG).
        Processes PMIDs in batches to respect API rate limits.
        """
        if not pmids:
            return

        for batch_start in range(0, len(pmids), self.batch_size):
            batch = pmids[batch_start : batch_start + self.batch_size]
            batch_num = batch_start // self.batch_size + 1
            log.info("Fetching batch %d (%d PMIDs)", batch_num, len(batch))

            try:
                data = self._get("efetch.fcgi", {
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "rettype": "abstract",
                    "retmode": "xml",
                })
            except Exception as exc:
                log.warning("Batch %d failed: %s — skipping", batch_num, exc)
                continue

            raw_articles = (
                data.get("PubmedArticleSet", {}).get("PubmedArticle", [])
            )
            if isinstance(raw_articles, dict):
                raw_articles = [raw_articles]

            for raw in raw_articles:
                article = self._parse_article(raw)
                if article is not None:
                    yield article

            time.sleep(self._delay)

    def search_and_fetch(
        self, query: str, max_results: int = 20
    ) -> list[Article]:
        """Convenience: search then fetch in one call."""
        pmids = self.search(query, max_results)
        return list(self.fetch_articles(pmids))

    # ── XML parsing ──────────────────────────────────────────────

    def _parse_article(self, raw: dict) -> Article | None:
        """
        Convert a raw xmltodict node into a clean :class:`Article`.
        Returns ``None`` if the article has no usable abstract.
        """
        try:
            medline: dict = raw.get("MedlineCitation", {})
            article: dict = medline.get("Article", {})

            # PMID
            pmid_raw = medline.get("PMID", {})
            pmid = str(pmid_raw.get("#text", pmid_raw) if isinstance(pmid_raw, dict) else pmid_raw)

            # Title — can be a plain string or dict with markup
            title_raw = article.get("ArticleTitle", "")
            title = title_raw.get("#text", "") if isinstance(title_raw, dict) else str(title_raw)

            # Abstract
            abstract = self._extract_abstract(
                article.get("Abstract", {}).get("AbstractText", "")
            )
            if not abstract or len(abstract) < 50:
                return None  # Not useful for RAG

            # Authors
            author_list = article.get("AuthorList", {}).get("Author", [])
            if isinstance(author_list, dict):
                author_list = [author_list]
            authors = [
                f"{a.get('ForeName', '')} {a.get('LastName', '')}".strip()
                for a in author_list
                if isinstance(a, dict) and a.get("LastName")
            ]

            # Journal and publication date
            journal_info = article.get("Journal", {})
            journal = str(journal_info.get("Title", "Unknown Journal"))
            pub_date_raw = (
                journal_info.get("JournalIssue", {}).get("PubDate", {})
            )
            pub_date = str(
                pub_date_raw.get("Year", pub_date_raw.get("MedlineDate", "Unknown"))
            )

            # DOI
            id_list = article.get("ELocationID", [])
            if isinstance(id_list, dict):
                id_list = [id_list]
            doi = next(
                (
                    item.get("#text", "")
                    for item in id_list
                    if isinstance(item, dict) and item.get("@EIdType") == "doi"
                ),
                "",
            )

            # Keywords
            kw_raw = medline.get("KeywordList", {}).get("Keyword", [])
            if isinstance(kw_raw, dict):
                kw_raw = [kw_raw]
            keywords = [
                (kw.get("#text", kw) if isinstance(kw, dict) else kw)
                for kw in kw_raw
                if kw
            ]

            # MeSH headings
            mesh_raw = medline.get("MeshHeadingList", {}).get("MeshHeading", [])
            if isinstance(mesh_raw, dict):
                mesh_raw = [mesh_raw]
            mesh_terms = [
                mh.get("DescriptorName", {}).get("#text", "")
                for mh in mesh_raw
                if isinstance(mh, dict)
            ]

            return Article(
                pmid=pmid,
                title=title.strip(),
                abstract=abstract.strip(),
                authors=authors,
                journal=journal.strip(),
                pub_date=pub_date.strip(),
                keywords=[k for k in keywords if k],
                mesh_terms=[m for m in mesh_terms if m],
                doi=doi.strip(),
            )

        except Exception as exc:  # pragma: no cover
            log.debug("Failed to parse article: %s", exc)
            return None

    def _extract_abstract(self, raw) -> str:
        """
        Handle three PubMed abstract formats:
        - Plain string: ``"Background text."``
        - Single dict (structured section): ``{"@Label": "RESULTS", "#text": "..."}``
        - List of dicts (multi-section structured abstract)
        """
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict):
            label = raw.get("@Label", "")
            text = raw.get("#text", "")
            return f"{label}: {text}" if label else text
        if isinstance(raw, list):
            parts: list[str] = []
            for section in raw:
                if isinstance(section, dict):
                    label = section.get("@Label", "")
                    text = section.get("#text", "")
                    parts.append(f"{label}: {text}" if label else text)
                elif isinstance(section, str):
                    parts.append(section)
            return " ".join(p for p in parts if p)
        return ""
