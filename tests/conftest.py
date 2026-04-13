"""
tests/conftest.py
──────────────────
Shared pytest fixtures for the MedRAG test suite.
All fixtures use mocks — no real API calls are made during testing.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.ingestion.pubmed_client import Article
from src.ingestion.openfda_client import DrugRecord


# ── Article fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def sample_article() -> Article:
    return Article(
        pmid="38234567",
        title="Metformin and Cardiovascular Outcomes in Type 2 Diabetes: A Systematic Review",
        abstract=(
            "Background: Metformin remains the first-line pharmacological treatment "
            "for type 2 diabetes. This systematic review examines cardiovascular outcomes "
            "across 45 randomised controlled trials with a combined cohort of 120,000 patients. "
            "Methods: MEDLINE, Embase, and Cochrane databases were searched from 2000 to 2024. "
            "Results: Metformin was associated with a 14% relative risk reduction in major "
            "adverse cardiovascular events (MACE) compared to sulfonylureas (RR 0.86, 95% CI "
            "0.78–0.95). Conclusions: Metformin demonstrates favourable cardiovascular effects "
            "beyond glycaemic control."
        ),
        authors=["Smith J", "Doe A", "Johnson B", "Williams C"],
        journal="Diabetes Care",
        pub_date="2024",
        keywords=["metformin", "type 2 diabetes", "cardiovascular outcomes", "MACE"],
        mesh_terms=["Diabetes Mellitus, Type 2", "Metformin", "Cardiovascular Diseases"],
        doi="10.2337/dc24-0001",
    )


@pytest.fixture
def sample_article_no_abstract() -> Article:
    return Article(
        pmid="99999999",
        title="Short communication without abstract",
        abstract="",
        authors=["Author A"],
        journal="Test Journal",
        pub_date="2023",
    )


@pytest.fixture
def article_many_authors() -> Article:
    return Article(
        pmid="11111111",
        title="Multi-author study",
        abstract="This study involved many researchers.",
        authors=["Alpha A", "Beta B", "Gamma G", "Delta D", "Epsilon E", "Zeta Z"],
        journal="Nature Medicine",
        pub_date="2024",
    )


@pytest.fixture
def sample_drug() -> DrugRecord:
    return DrugRecord(
        brand_name="Glucophage",
        generic_name="Metformin Hydrochloride",
        manufacturer="Bristol-Myers Squibb Company",
        indications=(
            "GLUCOPHAGE is indicated as an adjunct to diet and exercise to improve "
            "glycemic control in adults and pediatric patients 10 years of age and older "
            "with type 2 diabetes mellitus."
        ),
        contraindications=(
            "Renal impairment (eGFR below 30 mL/min/1.73 m²). "
            "Acute or chronic metabolic acidosis, including diabetic ketoacidosis."
        ),
        warnings=(
            "Lactic acidosis: Metformin can cause lactic acidosis, a rare but serious "
            "metabolic complication. Risk factors include renal impairment, hepatic impairment, "
            "congestive heart failure, and excessive alcohol intake."
        ),
        adverse_reactions=(
            "The most common adverse reactions (incidence ≥5%) are diarrhea, nausea/vomiting, "
            "flatulence, abdominal discomfort, and indigestion. These are dose-dependent and "
            "typically transient."
        ),
        dosage=(
            "Starting dose: 500 mg twice daily or 850 mg once daily, given with meals. "
            "Titrate in increments of 500 mg weekly or 850 mg every 2 weeks."
        ),
        drug_interactions=(
            "Cationic drugs (amiloride, digoxin, morphine, quinine, ranitidine, triamterene, "
            "trimethoprim, vancomycin) may compete for common renal tubular transport systems."
        ),
        mechanism_of_action=(
            "Metformin decreases hepatic glucose production, decreases intestinal absorption "
            "of glucose, and improves insulin sensitivity by increasing peripheral glucose "
            "uptake and utilisation."
        ),
        nda_number="NDA 020357",
    )
