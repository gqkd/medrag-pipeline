"""
tests/test_pipeline.py
───────────────────────
Full test suite for the MedRAG pipeline.

All external calls (OpenAI, PubMed, FDA) are mocked.
No network access is needed to run these tests.

Run:
    pytest tests/ -v
    pytest tests/ --cov=src --cov-report=term-missing
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

from src.ingestion.pubmed_client import Article, PubMedClient
from src.ingestion.openfda_client import DrugRecord, AdverseEventSummary, OpenFDAClient


# ═════════════════════════════════════════════════════════════════════════════
# Article dataclass
# ═════════════════════════════════════════════════════════════════════════════

class TestArticle:
    """Unit tests for the Article dataclass."""

    def test_full_text_contains_title_and_abstract(self, sample_article):
        text = sample_article.full_text
        assert "Title:" in text
        assert "Abstract:" in text
        assert sample_article.title in text
        assert sample_article.abstract[:50] in text

    def test_full_text_contains_keywords_when_present(self, sample_article):
        text = sample_article.full_text
        assert "Keywords:" in text
        assert "metformin" in text

    def test_full_text_no_keywords_section_when_empty(self, sample_article_no_abstract):
        # Article with no keywords
        sample_article_no_abstract.keywords = []
        text = sample_article_no_abstract.full_text
        assert "Keywords:" not in text

    def test_citation_format_with_multiple_authors(self, sample_article):
        citation = sample_article.citation
        assert "PMID: 38234567" in citation
        assert "2024" in citation
        assert "Smith J" in citation
        assert "et al." in citation  # 4 authors → truncate

    def test_citation_format_single_author(self):
        a = Article(
            pmid="12345", title="Single author paper",
            abstract="Abstract text here.",
            authors=["Solo A"], journal="Test", pub_date="2023",
        )
        assert "et al." not in a.citation
        assert "Solo A" in a.citation

    def test_citation_three_authors_no_et_al(self):
        a = Article(
            pmid="12345", title="Three author paper",
            abstract="Abstract.",
            authors=["A A", "B B", "C C"], journal="Test", pub_date="2023",
        )
        assert "et al." not in a.citation

    def test_citation_four_authors_adds_et_al(self, article_many_authors):
        assert "et al." in article_many_authors.citation

    def test_to_metadata_has_required_keys(self, sample_article):
        meta = sample_article.to_metadata()
        required = {"pmid", "title", "authors", "journal", "pub_date", "citation", "source"}
        assert required.issubset(meta.keys())

    def test_to_metadata_source_is_pubmed(self, sample_article):
        assert sample_article.to_metadata()["source"] == "pubmed"

    def test_to_metadata_authors_joined(self, sample_article):
        meta = sample_article.to_metadata()
        assert ";" in meta["authors"]
        assert "Smith J" in meta["authors"]

    def test_to_metadata_truncates_long_title(self):
        a = Article(
            pmid="1", title="A" * 300, abstract="Abstract.",
            authors=["A A"], journal="J", pub_date="2024",
        )
        assert len(a.to_metadata()["title"]) <= 200


# ═════════════════════════════════════════════════════════════════════════════
# DrugRecord dataclass
# ═════════════════════════════════════════════════════════════════════════════

class TestDrugRecord:
    def test_full_text_contains_key_sections(self, sample_drug):
        text = sample_drug.full_text
        assert "Metformin Hydrochloride" in text
        assert "Indications:" in text
        assert "Warnings:" in text
        assert "Contraindications:" in text

    def test_full_text_skips_empty_sections(self):
        dr = DrugRecord(
            brand_name="TestDrug", generic_name="testgeneric",
            manufacturer="TestCo",
            indications="Used for X.",
            contraindications="",  # empty
            warnings="",           # empty
            adverse_reactions="Nausea.",
            dosage="", drug_interactions="",
        )
        text = dr.full_text
        assert "Contraindications:" not in text
        assert "Warnings:" not in text
        assert "Indications:" in text

    def test_citation_includes_nda(self, sample_drug):
        assert "NDA 020357" in sample_drug.citation

    def test_citation_without_nda(self):
        dr = DrugRecord(
            brand_name="Test", generic_name="test",
            manufacturer="TestCo", indications="X",
            contraindications="", warnings="",
            adverse_reactions="", dosage="", drug_interactions="",
            nda_number="",
        )
        assert "—" not in dr.citation or "Test" in dr.citation

    def test_to_metadata_source(self, sample_drug):
        assert sample_drug.to_metadata()["source"] == "openfda"

    def test_to_metadata_keys(self, sample_drug):
        meta = sample_drug.to_metadata()
        assert {"brand_name", "generic_name", "nda_number", "citation", "source"} \
               .issubset(meta.keys())


# ═════════════════════════════════════════════════════════════════════════════
# AdverseEventSummary
# ═════════════════════════════════════════════════════════════════════════════

class TestAdverseEventSummary:
    def test_to_text_with_events(self):
        summary = AdverseEventSummary(
            drug_name="metformin",
            events=[
                {"term": "Diarrhea", "count": 15234},
                {"term": "Nausea", "count": 8901},
            ],
            total_reports=50000,
        )
        text = summary.to_text()
        assert "metformin" in text
        assert "Diarrhea" in text
        assert "15,234" in text
        assert "FAERS" in text

    def test_to_text_empty_events(self):
        summary = AdverseEventSummary(drug_name="unknowndrug", events=[])
        text = summary.to_text()
        assert "No adverse events found" in text
        assert "unknowndrug" in text


# ═════════════════════════════════════════════════════════════════════════════
# PubMedClient — abstract parsing
# ═════════════════════════════════════════════════════════════════════════════

class TestPubMedClientParsing:
    """Test the XML parsing logic without network calls."""

    def setup_method(self):
        # Patch settings to avoid requiring .env in CI
        with patch("src.config.Settings") as mock_settings:
            mock_settings.return_value.pubmed_email = "test@test.com"
            mock_settings.return_value.pubmed_api_key = ""
        self.client = PubMedClient.__new__(PubMedClient)
        self.client.email = "test@test.com"
        self.client.api_key = ""
        self.client._delay = 0.38
        self.client._base = {"email": "test@test.com", "tool": "medrag-test"}
        import requests
        self.client._session = requests.Session()

    def test_plain_string_abstract(self):
        result = self.client._extract_abstract("Simple abstract text.")
        assert result == "Simple abstract text."

    def test_structured_dict_abstract_with_label(self):
        raw = {"@Label": "BACKGROUND", "#text": "Background here."}
        result = self.client._extract_abstract(raw)
        assert "BACKGROUND:" in result
        assert "Background here." in result

    def test_structured_dict_abstract_no_label(self):
        raw = {"#text": "Plain content."}
        result = self.client._extract_abstract(raw)
        assert result == "Plain content."

    def test_multi_section_abstract(self):
        raw = [
            {"@Label": "BACKGROUND", "#text": "Background text."},
            {"@Label": "METHODS", "#text": "Methods described."},
            {"@Label": "RESULTS", "#text": "Key findings here."},
            {"@Label": "CONCLUSIONS", "#text": "We conclude this."},
        ]
        result = self.client._extract_abstract(raw)
        assert "BACKGROUND:" in result
        assert "METHODS:" in result
        assert "RESULTS:" in result
        assert "CONCLUSIONS:" in result
        assert "Background text." in result

    def test_list_with_plain_strings(self):
        raw = ["Part one.", "Part two.", "Part three."]
        result = self.client._extract_abstract(raw)
        assert "Part one." in result
        assert "Part three." in result

    def test_empty_dict(self):
        assert self.client._extract_abstract({}) == ""

    def test_empty_list(self):
        assert self.client._extract_abstract([]) == ""

    def test_none_returns_empty(self):
        # None is not a handled type → should return ""
        assert self.client._extract_abstract(None) == ""  # type: ignore


# ═════════════════════════════════════════════════════════════════════════════
# VectorStore
# ═════════════════════════════════════════════════════════════════════════════

class TestMedRAGVectorStore:
    """Test chunking logic and store interface without FAISS/OpenAI."""

    @patch("src.pipeline.vector_store.OpenAIEmbeddings")
    @patch("src.pipeline.vector_store.FAISS")
    def test_add_articles_returns_nonzero_count(
        self, mock_faiss_cls, mock_embeddings_cls, sample_article
    ):
        from src.pipeline.vector_store import MedRAGVectorStore

        mock_store = MagicMock()
        mock_faiss_cls.from_documents.return_value = mock_store

        vs = MedRAGVectorStore(index_path=Path("/tmp/test_vs"))
        count = vs.add_articles([sample_article])
        assert count >= 1
        assert mock_faiss_cls.from_documents.called

    @patch("src.pipeline.vector_store.OpenAIEmbeddings")
    def test_add_empty_articles_returns_zero(self, mock_embeddings_cls):
        from src.pipeline.vector_store import MedRAGVectorStore

        vs = MedRAGVectorStore(index_path=Path("/tmp/test_vs"))
        count = vs.add_articles([])
        assert count == 0

    @patch("src.pipeline.vector_store.OpenAIEmbeddings")
    @patch("src.pipeline.vector_store.FAISS")
    def test_add_drug_records_returns_nonzero_count(
        self, mock_faiss_cls, mock_embeddings_cls, sample_drug
    ):
        from src.pipeline.vector_store import MedRAGVectorStore

        mock_store = MagicMock()
        mock_faiss_cls.from_documents.return_value = mock_store

        vs = MedRAGVectorStore(index_path=Path("/tmp/test_vs"))
        count = vs.add_drug_records([sample_drug])
        assert count >= 1

    @patch("src.pipeline.vector_store.OpenAIEmbeddings")
    def test_get_stats_empty_store(self, mock_embeddings_cls):
        from src.pipeline.vector_store import MedRAGVectorStore

        vs = MedRAGVectorStore(index_path=Path("/tmp/test_vs_empty"))
        stats = vs.get_stats()
        assert stats["status"] == "empty"
        assert stats["total_vectors"] == 0

    @patch("src.pipeline.vector_store.OpenAIEmbeddings")
    @patch("src.pipeline.vector_store.FAISS")
    def test_get_stats_loaded_store(self, mock_faiss_cls, mock_embeddings_cls, sample_article):
        from src.pipeline.vector_store import MedRAGVectorStore

        mock_store = MagicMock()
        mock_store.index.ntotal = 42
        mock_faiss_cls.from_documents.return_value = mock_store

        vs = MedRAGVectorStore(index_path=Path("/tmp/test_vs_loaded"))
        vs.add_articles([sample_article])
        stats = vs.get_stats()
        assert stats["status"] == "loaded"
        assert stats["total_vectors"] == 42

    @patch("src.pipeline.vector_store.OpenAIEmbeddings")
    def test_require_store_raises_when_empty(self, mock_embeddings_cls):
        from src.pipeline.vector_store import MedRAGVectorStore

        vs = MedRAGVectorStore(index_path=Path("/tmp/test_vs"))
        with pytest.raises(RuntimeError, match="not initialized"):
            vs._require_store()

    @patch("src.pipeline.vector_store.OpenAIEmbeddings")
    def test_save_raises_when_no_store(self, mock_embeddings_cls):
        from src.pipeline.vector_store import MedRAGVectorStore

        vs = MedRAGVectorStore(index_path=Path("/tmp/test_vs"))
        with pytest.raises(RuntimeError, match="Nothing to save"):
            vs.save()

    @patch("src.pipeline.vector_store.OpenAIEmbeddings")
    def test_load_returns_false_when_index_missing(self, mock_embeddings_cls):
        from src.pipeline.vector_store import MedRAGVectorStore

        vs = MedRAGVectorStore(index_path=Path("/tmp/nonexistent_index_xyz"))
        assert vs.load() is False

    @patch("src.pipeline.vector_store.OpenAIEmbeddings")
    @patch("src.pipeline.vector_store.FAISS")
    def test_drug_labels_use_larger_chunks(
        self, mock_faiss_cls, mock_embeddings_cls, sample_drug
    ):
        """Drug label splitter should have larger chunk_size than abstract splitter."""
        from src.pipeline.vector_store import MedRAGVectorStore

        mock_store = MagicMock()
        mock_faiss_cls.from_documents.return_value = mock_store

        vs = MedRAGVectorStore(index_path=Path("/tmp/test_vs"))
        assert vs._label_splitter._chunk_size > vs._abstract_splitter._chunk_size


# ═════════════════════════════════════════════════════════════════════════════
# OpenFDAClient — parsing
# ═════════════════════════════════════════════════════════════════════════════

class TestOpenFDAClientParsing:
    def setup_method(self):
        self.client = OpenFDAClient.__new__(OpenFDAClient)
        self.client.api_key = ""
        self.client.timeout = 20
        import requests
        self.client._session = requests.Session()

    def test_parse_label_extracts_fields(self, sample_drug):
        raw = {
            "openfda": {
                "brand_name": ["Glucophage"],
                "generic_name": ["Metformin Hydrochloride"],
                "manufacturer_name": ["BMS"],
                "application_number": ["NDA020357"],
            },
            "indications_and_usage": ["Used for type 2 diabetes."],
            "warnings": ["Lactic acidosis risk."],
            "adverse_reactions": ["Diarrhea, nausea."],
            "drug_interactions": ["Cationic drugs."],
            "dosage_and_administration": ["500mg twice daily."],
        }
        record = self.client._parse_label(raw)
        assert record.brand_name == "Glucophage"
        assert record.generic_name == "Metformin Hydrochloride"
        assert "type 2 diabetes" in record.indications
        assert "Lactic acidosis" in record.warnings

    def test_parse_label_handles_missing_openfda(self):
        raw = {
            "indications_and_usage": ["Some indication."],
        }
        record = self.client._parse_label(raw)
        assert record.brand_name == "Unknown"
        assert record.generic_name == "Unknown"

    def test_first_helper_with_list(self):
        result = self.client._first(["value one", "value two"])
        assert result == "value one"

    def test_first_helper_with_string(self):
        result = self.client._first("direct string")
        assert result == "direct string"

    def test_first_helper_with_none(self):
        result = self.client._first(None)
        assert result == ""

    def test_first_helper_truncates_long_content(self):
        long_text = "x" * 3000
        result = self.client._first([long_text])
        assert len(result) <= 2000


# ═════════════════════════════════════════════════════════════════════════════
# Integration: agent tool wiring
# ═════════════════════════════════════════════════════════════════════════════

class TestToolBuilding:
    @patch("src.pipeline.vector_store.OpenAIEmbeddings")
    @patch("src.pipeline.vector_store.FAISS")
    def test_build_tools_returns_four_tools(
        self, mock_faiss_cls, mock_embeddings_cls
    ):
        from src.pipeline.vector_store import MedRAGVectorStore
        from src.ingestion.openfda_client import OpenFDAClient
        from src.agent.tools import build_tools

        mock_store = MagicMock()
        mock_faiss_cls.from_documents.return_value = mock_store

        vs = MedRAGVectorStore(index_path=Path("/tmp/test_tools"))
        fda = OpenFDAClient.__new__(OpenFDAClient)

        tools = build_tools(vs, fda)
        assert len(tools) == 4

    @patch("src.pipeline.vector_store.OpenAIEmbeddings")
    @patch("src.pipeline.vector_store.FAISS")
    def test_tool_names(self, mock_faiss_cls, mock_embeddings_cls):
        from src.pipeline.vector_store import MedRAGVectorStore
        from src.ingestion.openfda_client import OpenFDAClient
        from src.agent.tools import build_tools

        vs = MedRAGVectorStore(index_path=Path("/tmp/test_tools"))
        fda = OpenFDAClient.__new__(OpenFDAClient)

        tools = build_tools(vs, fda)
        names = {t.name for t in tools}
        expected = {
            "search_pubmed_literature",
            "lookup_fda_drug_info",
            "search_drug_in_literature",
            "get_adverse_event_statistics",
        }
        assert names == expected

    @patch("src.pipeline.vector_store.OpenAIEmbeddings")
    @patch("src.pipeline.vector_store.FAISS")
    def test_all_tools_have_descriptions(self, mock_faiss_cls, mock_embeddings_cls):
        from src.pipeline.vector_store import MedRAGVectorStore
        from src.ingestion.openfda_client import OpenFDAClient
        from src.agent.tools import build_tools

        vs = MedRAGVectorStore(index_path=Path("/tmp/test_tools"))
        fda = OpenFDAClient.__new__(OpenFDAClient)

        tools = build_tools(vs, fda)
        for tool in tools:
            assert tool.description, f"Tool {tool.name} has no description"
            # Descriptions should be substantial (key for agent quality)
            assert len(tool.description) > 100, \
                f"Tool {tool.name} description is too short ({len(tool.description)} chars)"

    @patch("src.pipeline.vector_store.OpenAIEmbeddings")
    def test_search_pubmed_tool_handles_empty_store(self, mock_embeddings_cls):
        from src.pipeline.vector_store import MedRAGVectorStore
        from src.ingestion.openfda_client import OpenFDAClient
        from src.agent.tools import build_tools

        vs = MedRAGVectorStore(index_path=Path("/tmp/test_empty_store"))
        # Don't add any documents — store is empty
        fda = OpenFDAClient.__new__(OpenFDAClient)
        tools = build_tools(vs, fda)

        search_tool = next(t for t in tools if t.name == "search_pubmed_literature")
        result = search_tool.invoke("metformin diabetes")
        # Should return error message, not raise
        assert isinstance(result, str)
        assert len(result) > 0
