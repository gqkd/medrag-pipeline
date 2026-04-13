"""
scripts/query_agent.py
───────────────────────
CLI interface for querying the MedRAG agent.

Supports three modes:
  1. Single question:  python scripts/query_agent.py "your question"
  2. Interactive REPL: python scripts/query_agent.py --interactive
  3. Verbose mode:     python scripts/query_agent.py "question" --verbose

Usage::

    python scripts/query_agent.py "What are the side effects of metformin?"
    python scripts/query_agent.py --verbose "Compare GLP-1 agonists for T2D"
    python scripts/query_agent.py --interactive
    python scripts/query_agent.py --model gpt-4.1 "Complex multi-hop question"
"""

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

# ── Imports ──────────────────────────────────────────────────────────────────
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.table import Table

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_agent(model: str | None = None, verbose: bool = False):
    """Load agent with spinner. Raises RuntimeError if index not found."""
    from src.agent.agent import build_agent
    return build_agent(model=model, verbose=verbose)


def _print_response(question: str, response, elapsed: float) -> None:
    """Pretty-print a single AgentResponse to the terminal."""
    # Answer
    console.print()
    console.print(Rule(f"[bold green]Answer", style="green"))
    console.print(Markdown(response.answer))

    # Sources
    if response.sources:
        console.print()
        console.print(Rule("[bold blue]Sources", style="blue dim"))
        for i, src in enumerate(response.sources, 1):
            icon = "📄" if "PMID" in src else "💊"
            console.print(f"  [blue][{i}][/]  {icon}  {src}")

    # Metadata footer
    console.print()
    tools_str = ", ".join(sorted(set(response.tools_used))) or "none"
    console.print(
        f"[dim]⏱  {elapsed:.1f}s  ·  🔧 {tools_str}[/dim]"
    )
    console.print()


def _print_reasoning(response) -> None:
    """Print the agent's reasoning trace (verbose mode)."""
    if not response.intermediate_steps:
        return
    console.print(Rule("[bold yellow]Reasoning Trace", style="yellow dim"))
    for i, (action, observation) in enumerate(response.intermediate_steps, 1):
        console.print(f"\n[yellow]Step {i}[/]")
        if hasattr(action, "tool"):
            console.print(f"  [cyan]Tool:[/]  {action.tool}")
        if hasattr(action, "tool_input"):
            console.print(f"  [cyan]Input:[/] {action.tool_input}")
        obs_preview = str(observation)[:300].replace("\n", " ")
        console.print(f"  [dim]Obs:   {obs_preview}…[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────


def single_query(
    question: str,
    model: str | None = None,
    verbose: bool = False,
) -> int:
    """Run a single query and print the result. Returns exit code."""
    with console.status("[bold green]Loading MedRAG agent..."):
        try:
            agent = _load_agent(model=model, verbose=verbose)
        except RuntimeError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            return 1

    console.print(f"\n[bold cyan]Question:[/] {question}\n")

    with console.status("[bold green]Agent reasoning..."):
        t0 = time.perf_counter()
        try:
            response = agent.query(question)
        except Exception as exc:
            console.print(f"[red]Agent error:[/red] {exc}")
            return 1
        elapsed = time.perf_counter() - t0

    if verbose:
        _print_reasoning(response)

    _print_response(question, response, elapsed)
    return 0


def interactive_session(
    model: str | None = None,
    verbose: bool = False,
) -> int:
    """
    Start an interactive REPL for multi-turn querying.
    Type 'exit', 'quit', or press Ctrl-C to end the session.
    """
    console.print(Panel.fit(
        "[bold cyan]MedRAG Interactive Session[/]\n\n"
        "Ask any biomedical question. Type [bold red]exit[/] to quit.\n"
        f"Model: [yellow]{model or 'default'}[/]",
        title="🧬 MedRAG",
        border_style="cyan",
    ))

    # Load agent once for the whole session
    with console.status("[bold green]Loading agent..."):
        try:
            agent = _load_agent(model=model, verbose=verbose)
        except RuntimeError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            return 1

    vs_stats = agent.vector_store.get_stats()
    console.print(
        f"[green]✓ Agent ready[/]  ·  "
        f"[dim]{vs_stats.get('total_vectors', 0):,} vectors indexed[/dim]\n"
    )

    session_stats = {"queries": 0, "total_time": 0.0, "total_sources": 0}

    while True:
        try:
            question = console.input("[bold yellow]You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q", ":q"):
            break
        if question.lower() == "stats":
            _print_session_stats(session_stats)
            continue
        if question.lower() == "help":
            _print_help()
            continue

        with console.status("[bold green]Thinking..."):
            t0 = time.perf_counter()
            try:
                response = agent.query(question)
            except Exception as exc:
                console.print(f"[red]Error:[/red] {exc}\n")
                continue
            elapsed = time.perf_counter() - t0

        session_stats["queries"] += 1
        session_stats["total_time"] += elapsed
        session_stats["total_sources"] += len(response.sources)

        if verbose:
            _print_reasoning(response)

        console.print(f"\n[bold green]MedRAG:[/]")
        console.print(Markdown(response.answer))
        if response.sources:
            console.print(
                f"\n[dim]Sources: "
                + " | ".join(f"[{i+1}] {s[:60]}" for i, s in enumerate(response.sources[:3]))
                + "[/dim]"
            )
        console.print(f"[dim]⏱ {elapsed:.1f}s  ·  🔧 {', '.join(set(response.tools_used))}[/dim]\n")

    _print_session_stats(session_stats)
    console.print("[cyan]Goodbye![/]")
    return 0


def _print_session_stats(stats: dict) -> None:
    if stats["queries"] == 0:
        return
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Label", style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Queries", str(stats["queries"]))
    table.add_row("Avg time", f"{stats['total_time'] / stats['queries']:.1f}s")
    table.add_row("Total sources cited", str(stats["total_sources"]))
    console.print(Panel(table, title="[dim]Session stats", border_style="dim"))


def _print_help() -> None:
    console.print(
        "\n[bold]Commands:[/]\n"
        "  [cyan]stats[/]  — show session statistics\n"
        "  [cyan]help[/]   — show this message\n"
        "  [cyan]exit[/]   — end the session\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="query_agent",
        description="Query the MedRAG biomedical research agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Biomedical question to answer (omit to start interactive mode)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive REPL session",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show agent reasoning trace (Thought/Action/Observation steps)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        metavar="MODEL",
        help="Override LLM model (e.g. gpt-4.1 for harder questions)",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.interactive or not args.question:
        return interactive_session(model=args.model, verbose=args.verbose)
    return single_query(
        question=args.question,
        model=args.model,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
