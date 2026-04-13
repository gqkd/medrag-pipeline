"""
scripts/run_pipeline.py
────────────────────────
CLI entry point for the MedRAG ETL pipeline.

Orchestrates the full Extract → Transform → Load flow:
  1. Extract: fetch PubMed articles and FDA drug records via their APIs
  2. Transform: chunk text into overlapping segments
  3. Load: embed with OpenAI and persist a FAISS index to disk

Usage::

    # Basic ingest
    python scripts/run_pipeline.py --query "type 2 diabetes treatment"

    # With drug label data
    python scripts/run_pipeline.py \\
        --query "diabetes mellitus pharmacotherapy" \\
        --max_results 50 \\
        --drugs metformin semaglutide liraglutide empagliflozin

    # Append to an existing index (don't overwrite)
    python scripts/run_pipeline.py \\
        --query "hypertension ACE inhibitors" \\
        --drugs lisinopril amlodipine \\
        --append

    # Quiet mode (no rich output)
    python scripts/run_pipeline.py --query "cancer immunotherapy" --quiet
"""

from __future__ import annotations

import sys
import logging
import argparse
from pathlib import Path

# ── Path setup (run from any directory) ─────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

# ── Imports after sys.path is set ───────────────────────────────────────────
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from src.ingestion.pubmed_client import PubMedClient, Article
from src.ingestion.openfda_client import OpenFDAClient, DrugRecord
from src.pipeline.vector_store import MedRAGVectorStore

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline function (also importable from other modules)
# ─────────────────────────────────────────────────────────────────────────────


def run_pipeline(
    query: str,
    max_results: int = 30,
    drug_names: list[str] | None = None,
    append: bool = False,
    quiet: bool = False,
) -> dict:
    """
    Execute the full ETL pipeline.

    Args:
        query:       PubMed search query.
        max_results: Maximum number of PubMed articles to ingest.
        drug_names:  List of drug names to fetch FDA labels for.
        append:      If True, merge into existing FAISS index.
        quiet:       Suppress rich console output.

    Returns:
        Summary dict with keys: articles_fetched, drug_records, chunks_indexed.
    """
    drug_names = drug_names or []
    stats = {"articles_fetched": 0, "drug_records": 0, "chunks_indexed": 0}

    if not quiet:
        console.print(Panel.fit(
            f"[bold cyan]MedRAG ETL Pipeline[/bold cyan]\n\n"
            f"  Query        : [yellow]{query}[/]\n"
            f"  Max articles : [yellow]{max_results}[/]\n"
            f"  Drugs        : [yellow]{', '.join(drug_names) or 'none'}[/]\n"
            f"  Mode         : [yellow]{'append' if append else 'overwrite'}[/]",
            title="🧬 Starting Pipeline",
            border_style="cyan",
        ))

    pubmed = PubMedClient()
    fda = OpenFDAClient()
    vs = MedRAGVectorStore()

    # Load existing index if appending
    if append:
        loaded = vs.load()
        if not loaded and not quiet:
            console.print(
                "[yellow]⚠  No existing index found — will create a new one.[/]"
            )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TimeElapsedColumn(),
        console=console,
        disable=quiet,
    ) as progress:

        # ── STEP 1: PubMed extraction ─────────────────────────────────
        task_pubmed = progress.add_task(
            "[cyan]Fetching PubMed articles...", total=None
        )
        articles: list[Article] = pubmed.search_and_fetch(
            query, max_results=max_results
        )
        stats["articles_fetched"] = len(articles)
        progress.update(
            task_pubmed,
            description=f"[green]✓ {len(articles)} PubMed articles fetched",
            completed=1,
            total=1,
        )

        if not articles and not quiet:
            console.print(
                "[yellow]⚠  No PubMed articles found. "
                "Try a broader query or check your internet connection.[/]"
            )

        # ── STEP 2: FDA label extraction ──────────────────────────────
        drug_records: list[DrugRecord] = []
        if drug_names:
            task_fda = progress.add_task(
                "[cyan]Fetching FDA drug labels...", total=len(drug_names)
            )
            for drug in drug_names:
                record = fda.get_drug_label(drug)
                if record:
                    drug_records.append(record)
                    if not quiet:
                        console.print(
                            f"  [green]✓[/] {record.brand_name} ({record.generic_name})"
                        )
                else:
                    if not quiet:
                        console.print(f"  [yellow]–[/] '{drug}' not found in FDA database")
                progress.advance(task_fda)

            stats["drug_records"] = len(drug_records)
            progress.update(
                task_fda,
                description=f"[green]✓ {len(drug_records)}/{len(drug_names)} FDA records fetched",
            )

        # ── STEP 3: Embed + index ─────────────────────────────────────
        task_embed = progress.add_task(
            "[cyan]Embedding and indexing...", total=None
        )
        if articles:
            chunks = vs.add_articles(articles)
            stats["chunks_indexed"] += chunks

        if drug_records:
            drug_chunks = vs.add_drug_records(drug_records)
            stats["chunks_indexed"] += drug_chunks

        progress.update(
            task_embed,
            description=f"[green]✓ {stats['chunks_indexed']} chunks embedded",
            completed=1,
            total=1,
        )

        # ── STEP 4: Persist ───────────────────────────────────────────
        task_save = progress.add_task("[cyan]Saving vector index...", total=None)
        if stats["chunks_indexed"] > 0:
            vs.save()
        progress.update(
            task_save,
            description=f"[green]✓ Saved → {vs.index_path}",
            completed=1,
            total=1,
        )

    # ── Summary table ─────────────────────────────────────────────
    if not quiet:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Label", style="dim")
        table.add_column("Value", style="bold")
        table.add_row("📄 Articles ingested", str(stats["articles_fetched"]))
        table.add_row("💊 Drug records", str(stats["drug_records"]))
        table.add_row("🔢 Vector chunks", str(stats["chunks_indexed"]))
        table.add_row("💾 Index path", str(vs.index_path))
        console.print(Panel(table, title="[bold green]✅ Pipeline Complete", border_style="green"))

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="MedRAG ETL Pipeline — ingest PubMed + FDA data into a FAISS index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="PubMed search query (supports full PubMed syntax)",
    )
    parser.add_argument(
        "--max_results", "-n",
        type=int,
        default=30,
        metavar="N",
        help="Max number of PubMed articles to ingest (default: 30, max: 200)",
    )
    parser.add_argument(
        "--drugs", "-d",
        nargs="*",
        default=[],
        dest="drug_names",
        metavar="DRUG",
        help="Drug names to fetch FDA label data for",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing index instead of overwriting",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress rich console output",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        stats = run_pipeline(
            query=args.query,
            max_results=args.max_results,
            drug_names=args.drug_names,
            append=args.append,
            quiet=args.quiet,
        )
        return 0 if stats["chunks_indexed"] > 0 else 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user.[/]")
        return 130
    except Exception as exc:
        console.print(f"[red]Pipeline error: {exc}[/]")
        logging.debug("Traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
