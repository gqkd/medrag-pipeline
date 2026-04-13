"""
src/agent/tools.py
───────────────────
LangChain tools exposed to the MedRAG ReAct agent.

Design principle: the tool DESCRIPTION is the contract between the LLM and
the tool. The agent decides which tool to call based solely on these strings.
Keep descriptions precise, specific, and honest about limitations.

Tools are built via a factory function so dependencies (vector store, FDA
client) are injected at runtime rather than held as module-level globals.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from src.pipeline.vector_store import MedRAGVectorStore
    from src.ingestion.openfda_client import OpenFDAClient

log = logging.getLogger(__name__)


def build_tools(
    vector_store: "MedRAGVectorStore",
    fda_client: "OpenFDAClient",
) -> list:
    """
    Return all agent tools with their dependencies injected.
    Call once during agent initialisation.
    """

    # ── Tool 1: PubMed literature search ─────────────────────────────────

    @tool
    def search_pubmed_literature(query: str) -> str:
        """
        Search the indexed PubMed biomedical literature and return relevant
        article excerpts with full citation information.

        USE THIS TOOL FOR:
        - Questions about diseases, medical conditions, or clinical symptoms
        - Evidence for treatments, therapies, or clinical interventions
        - Pharmacological mechanisms of drugs (how a drug works)
        - Clinical trial results, efficacy statistics, outcome data
        - Epidemiology: prevalence, incidence, risk factors
        - Pathophysiology and disease mechanisms
        - Comparative effectiveness between treatments

        DO NOT use for official drug approval status or legal warnings
        (use lookup_fda_drug_info for those).

        Args:
            query: A precise medical search query. Include drug names, conditions,
                   and the specific aspect you need (e.g. mechanism, safety, efficacy).
                   Good examples:
                   - "metformin HbA1c reduction type 2 diabetes randomized trial"
                   - "GLP-1 receptor agonist cardiovascular outcomes MACE"
                   - "BRCA1 BRCA2 mutation hereditary breast cancer risk"
                   - "COVID-19 long term neurological symptoms post-acute sequelae"

        Returns:
            Formatted list of relevant article excerpts with PMID citations.
        """
        log.info("Tool: search_pubmed_literature | query=%r", query)
        try:
            docs = vector_store.similarity_search(query, k=4)
        except RuntimeError as exc:
            return f"Vector store error: {exc}"

        if not docs:
            return (
                f"No results found in the indexed literature for: '{query}'.\n"
                "Try a broader query, different terminology, or check if the "
                "ETL pipeline has been run for this topic."
            )

        results: list[str] = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            citation = meta.get("citation", "Unknown source")
            journal = meta.get("journal", "")
            pub_date = meta.get("pub_date", "")
            content = doc.page_content.strip()
            results.append(
                f"[{i}] {citation}\n"
                f"    {journal}{' (' + pub_date + ')' if pub_date else ''}\n"
                f"    {content[:450]}{'…' if len(content) > 450 else ''}"
            )

        header = f"Found {len(docs)} relevant article(s):\n"
        return header + "\n\n".join(results)

    # ── Tool 2: FDA drug label lookup ─────────────────────────────────────

    @tool
    def lookup_fda_drug_info(drug_name: str) -> str:
        """
        Retrieve official FDA drug label information for a specific drug.
        Data comes live from the OpenFDA API (not from the indexed literature).

        USE THIS TOOL FOR:
        - Official FDA-approved indications (what the drug is approved to treat)
        - Black box warnings and contraindications
        - Listed side effects and adverse reactions
        - Drug-drug interactions listed in the official label
        - Dosage and administration guidelines
        - Manufacturer information and NDA/BLA numbers

        DO NOT use for research evidence, clinical trial data, or off-label uses
        (use search_pubmed_literature for those).

        Args:
            drug_name: Generic or brand name of the drug. Both work.
                       Examples: "metformin", "semaglutide", "Ozempic",
                                 "lisinopril", "aspirin", "Humira"

        Returns:
            Structured FDA label information with source citation.
        """
        log.info("Tool: lookup_fda_drug_info | drug=%r", drug_name)
        try:
            record = fda_client.get_drug_label(drug_name)
        except Exception as exc:
            return f"FDA API error for '{drug_name}': {exc}"

        if record is None:
            # Try fallback: search the locally indexed FDA data
            try:
                docs = vector_store.similarity_search(
                    f"{drug_name} drug indications warnings", k=3
                )
                fda_docs = [
                    d for d in docs if d.metadata.get("source") == "openfda"
                ]
                if fda_docs:
                    fallback = "\n\n".join(d.page_content for d in fda_docs)
                    return (
                        f"Live FDA label not found for '{drug_name}'. "
                        f"Related indexed FDA data:\n\n{fallback}"
                    )
            except RuntimeError:
                pass
            return (
                f"No FDA label found for '{drug_name}'. "
                "The drug may not be in the FDA database, the spelling may differ, "
                "or it may be a non-US product."
            )

        sections: list[str] = [
            f"💊 Drug: {record.brand_name} ({record.generic_name})",
            f"🏭 Manufacturer: {record.manufacturer}",
        ]
        if record.indications:
            sections.append(f"✅ Indications:\n{record.indications[:700]}")
        if record.contraindications:
            sections.append(f"🚫 Contraindications:\n{record.contraindications[:500]}")
        if record.warnings:
            sections.append(f"⚠️  Warnings:\n{record.warnings[:600]}")
        if record.adverse_reactions:
            sections.append(f"🔴 Adverse Reactions:\n{record.adverse_reactions[:600]}")
        if record.drug_interactions:
            sections.append(f"🔄 Drug Interactions:\n{record.drug_interactions[:500]}")
        if record.mechanism_of_action:
            sections.append(f"⚗️  Mechanism of Action:\n{record.mechanism_of_action[:400]}")
        sections.append(f"📋 Source: {record.citation}")

        return "\n\n".join(sections)

    # ── Tool 3: Drug-specific literature search ───────────────────────────

    @tool
    def search_drug_in_literature(drug_name: str) -> str:
        """
        Search the indexed biomedical literature specifically for clinical
        research about a named drug.

        USE THIS TOOL FOR:
        - Clinical trial results for a specific drug (efficacy, safety data)
        - Comparative effectiveness studies
        - Long-term outcomes research
        - Meta-analyses and systematic reviews about a drug
        - Off-label use evidence
        - Recent research not reflected in the FDA label

        This is complementary to lookup_fda_drug_info — use both for complete
        drug questions (FDA label = regulatory, literature = evidence).

        Args:
            drug_name: Generic or brand name of the drug.
                       Examples: "metformin", "semaglutide", "atorvastatin"

        Returns:
            Relevant publication excerpts with PMID citations.
        """
        log.info("Tool: search_drug_in_literature | drug=%r", drug_name)
        query = f"{drug_name} clinical efficacy safety outcomes randomized trial"
        try:
            docs = vector_store.similarity_search(query, k=4)
        except RuntimeError as exc:
            return f"Vector store error: {exc}"

        if not docs:
            return f"No clinical literature found for '{drug_name}' in the index."

        results: list[str] = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            citation = meta.get("citation", "Unknown source")
            content = doc.page_content.strip()
            results.append(
                f"[{i}] {citation}\n"
                f"    {content[:450]}{'…' if len(content) > 450 else ''}"
            )

        return f"Clinical literature for '{drug_name}':\n\n" + "\n\n".join(results)

    # ── Tool 4: FAERS adverse event statistics ────────────────────────────

    @tool
    def get_adverse_event_statistics(drug_name: str) -> str:
        """
        Retrieve real-world adverse event statistics from the FDA Adverse
        Event Reporting System (FAERS). Shows the most commonly reported
        side effects based on voluntary post-market surveillance reports.

        USE THIS TOOL FOR:
        - Real-world safety signals beyond controlled clinical trial data
        - Comparing severity of side effects across patient populations
        - When asked specifically about reported adverse events or FAERS data
        - Safety questions where real-world evidence is more important than labels

        Important caveat: FAERS reports are voluntary and may reflect reporting
        bias. High counts do not necessarily imply causation.

        Args:
            drug_name: Generic or brand name of the drug.
                       Examples: "metformin", "Ozempic", "warfarin"

        Returns:
            Ranked list of most-reported adverse events with counts.
        """
        log.info("Tool: get_adverse_event_statistics | drug=%r", drug_name)
        try:
            summary = fda_client.get_adverse_event_summary(drug_name, top_n=15)
        except Exception as exc:
            return f"FAERS lookup error for '{drug_name}': {exc}"

        return summary.to_text()

    # ─────────────────────────────────────────────────────────────────────
    return [
        search_pubmed_literature,
        lookup_fda_drug_info,
        search_drug_in_literature,
        get_adverse_event_statistics,
    ]
