"""
src/ingestion/openfda_client.py
────────────────────────────────
ETL — Extract layer: OpenFDA REST API client.

Covers three endpoints:
  • /drug/label   — official FDA drug labels (indications, warnings, interactions)
  • /drug/event   — adverse event reports from FAERS post-market surveillance
  • /drug/ndc     — National Drug Code directory (manufacturer info)

Public API docs: https://open.fda.gov/apis/
No authentication required; free API key raises rate limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

log = logging.getLogger(__name__)

_FDA_BASE = "https://api.fda.gov/drug"
_RETRY_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


# ─────────────────────────────────────────────────────────────────────────────
# Domain model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DrugRecord:
    """
    A structured FDA drug label record.
    Mirrors the key sections of an official drug label (package insert).
    """

    brand_name: str
    generic_name: str
    manufacturer: str
    indications: str
    contraindications: str
    warnings: str
    adverse_reactions: str
    dosage: str
    drug_interactions: str
    mechanism_of_action: str = ""
    nda_number: str = ""
    source: str = "openfda"

    @property
    def full_text(self) -> str:
        """
        Concatenated key sections for embedding.
        Sections with no content are excluded automatically.
        """
        sections = {
            "Drug": f"{self.brand_name} ({self.generic_name})",
            "Manufacturer": self.manufacturer,
            "Indications": self.indications[:1200],
            "Contraindications": self.contraindications[:600],
            "Warnings": self.warnings[:800],
            "Adverse Reactions": self.adverse_reactions[:800],
            "Drug Interactions": self.drug_interactions[:600],
            "Mechanism of Action": self.mechanism_of_action[:400],
        }
        return "\n\n".join(
            f"{label}: {content}"
            for label, content in sections.items()
            if content and content.strip()
        )

    @property
    def citation(self) -> str:
        return (
            f"FDA Drug Label — {self.brand_name} ({self.generic_name})"
            f"{' — ' + self.nda_number if self.nda_number else ''}"
        )

    def to_metadata(self) -> dict[str, str]:
        return {
            "brand_name": self.brand_name,
            "generic_name": self.generic_name,
            "manufacturer": self.manufacturer,
            "nda_number": self.nda_number,
            "citation": self.citation,
            "source": self.source,
        }


@dataclass
class AdverseEventSummary:
    """Top adverse events for a drug from the FAERS database."""

    drug_name: str
    events: list[dict[str, Any]]  # [{"term": str, "count": int}]
    total_reports: int = 0

    def to_text(self) -> str:
        if not self.events:
            return f"No adverse events found for {self.drug_name} in FAERS."
        lines = [
            f"Top reported adverse events for {self.drug_name} (FDA FAERS database):",
        ]
        for i, ev in enumerate(self.events[:15], 1):
            term = ev.get("term", "Unknown")
            count = ev.get("count", 0)
            lines.append(f"  {i:2d}. {term}: {count:,} reports")
        lines.append("Source: FDA Adverse Event Reporting System (FAERS)")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────


class OpenFDAClient:
    """
    REST client for the OpenFDA API.

    Usage::

        client = OpenFDAClient()
        record = client.get_drug_label("metformin")
        if record:
            print(record.indications)

        events = client.get_adverse_event_summary("metformin", top_n=10)
        print(events.to_text())
    """

    def __init__(self, api_key: str | None = None, timeout: int = 20) -> None:
        from src.config import settings

        self.api_key = api_key or settings.openfda_api_key or ""
        self.timeout = timeout
        self._session = requests.Session()

    # ── Private helpers ──────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        reraise=True,
    )
    def _get(self, endpoint: str, params: dict) -> dict:
        url = f"{_FDA_BASE}/{endpoint}.json"
        if self.api_key:
            params = {**params, "api_key": self.api_key}
        resp = self._session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _first(self, data: Any, fallback: str = "") -> str:
        """Safely extract the first element of an FDA list field."""
        if isinstance(data, list) and data:
            return str(data[0])[:2000]
        if isinstance(data, str):
            return data[:2000]
        return fallback

    # ── Drug label ───────────────────────────────────────────────

    def get_drug_label(self, drug_name: str) -> DrugRecord | None:
        """
        Fetch the FDA drug label for the given generic or brand name.
        Returns ``None`` if the drug is not found in the FDA database.

        The query searches both brand and generic name fields so that
        ``"Ozempic"`` and ``"semaglutide"`` both return the same record.
        """
        log.info("FDA label lookup: %r", drug_name)
        query = (
            f'openfda.generic_name:"{drug_name}" '
            f'OR openfda.brand_name:"{drug_name}"'
        )
        try:
            data = self._get("label", {"search": query, "limit": 1})
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                log.info("FDA label not found for %r", drug_name)
                return None
            raise

        results = data.get("results", [])
        if not results:
            return None

        return self._parse_label(results[0])

    def search_labels(self, query: str, limit: int = 5) -> list[DrugRecord]:
        """
        Free-text search across FDA drug labels.
        Returns up to ``limit`` :class:`DrugRecord` objects.

        Example::

            records = client.search_labels("GLP-1 receptor agonist diabetes")
        """
        log.info("FDA label search: %r (limit=%d)", query, limit)
        try:
            data = self._get("label", {"search": query, "limit": limit})
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return []
            raise
        return [self._parse_label(r) for r in data.get("results", [])]

    # ── Adverse events ───────────────────────────────────────────

    def get_adverse_event_summary(
        self, drug_name: str, top_n: int = 15
    ) -> AdverseEventSummary:
        """
        Return the most commonly reported adverse reactions for a drug
        from the FDA Adverse Event Reporting System (FAERS).

        Results are aggregated by MedDRA Preferred Term and sorted by
        report count (most frequent first).
        """
        log.info("FAERS lookup: %r", drug_name)
        try:
            data = self._get("event", {
                "search": f'patient.drug.medicinalproduct:"{drug_name}"',
                "count": "patient.reaction.reactionmeddrapt.exact",
                "limit": top_n,
            })
            events = data.get("results", [])
            total = data.get("meta", {}).get("results", {}).get("total", 0)
            return AdverseEventSummary(
                drug_name=drug_name, events=events, total_reports=total
            )
        except requests.HTTPError:
            return AdverseEventSummary(drug_name=drug_name, events=[])

    # ── XML/JSON parsing ─────────────────────────────────────────

    def _parse_label(self, raw: dict) -> DrugRecord:
        """
        Map a raw FDA label JSON object to a :class:`DrugRecord`.
        All fields are capped to prevent enormous embeddings.
        """
        fda = raw.get("openfda", {})

        brand_names: list = fda.get("brand_name", [])
        generic_names: list = fda.get("generic_name", [])
        manufacturers: list = fda.get("manufacturer_name", [])
        app_numbers: list = fda.get("application_number", [])

        return DrugRecord(
            brand_name=brand_names[0].title() if brand_names else "Unknown",
            generic_name=generic_names[0].title() if generic_names else "Unknown",
            manufacturer=manufacturers[0] if manufacturers else "Unknown",
            nda_number=app_numbers[0] if app_numbers else "",
            indications=self._first(raw.get("indications_and_usage")),
            contraindications=self._first(raw.get("contraindications")),
            warnings=(
                self._first(raw.get("warnings"))
                or self._first(raw.get("warnings_and_cautions"))
                or self._first(raw.get("boxed_warning"))
            ),
            adverse_reactions=self._first(raw.get("adverse_reactions")),
            dosage=self._first(raw.get("dosage_and_administration")),
            drug_interactions=self._first(raw.get("drug_interactions")),
            mechanism_of_action=self._first(
                raw.get("clinical_pharmacology")
                or raw.get("mechanism_of_action")
            ),
        )
