"""
src/agent/agent.py
───────────────────
MedRAG ReAct agent — LangGraph implementation.

Uses langgraph.prebuilt.create_react_agent which drives tool-calling via the
LLM's native function-calling API instead of text parsing. This is more
reliable, supports streaming out of the box, and integrates cleanly with
LangSmith tracing.

The public interface (AgentResponse, MedRAGAgent, build_agent) is unchanged
from the previous implementation — only the internal wiring changed.

Usage::

    from src.agent.agent import build_agent
    agent = build_agent()
    response = agent.query("What are the cardiovascular effects of metformin?")
    print(response)

    # Streaming
    for chunk in agent.stream_query("What are the side effects of semaglutide?"):
        if isinstance(chunk, str):
            print(chunk, end="", flush=True)
        else:
            print("\\nSources:", chunk.sources)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.agent.tools import build_tools
from src.agent.prompts import SYSTEM_PROMPT
from src.pipeline.vector_store import MedRAGVectorStore
from src.ingestion.openfda_client import OpenFDAClient

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Response model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AgentResponse:
    """
    Structured output from a single MedRAG query.

    Attributes:
        answer:              Final answer text from the LLM.
        sources:             Deduplicated source citation strings.
        tools_used:          Names of every tool invoked (may contain duplicates).
        processing_time_s:   Wall-clock seconds for the full agent run.
        intermediate_steps:  Raw LangGraph messages list for debugging.
    """

    answer: str
    sources: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    processing_time_s: float = 0.0
    intermediate_steps: list = field(default_factory=list)

    def __str__(self) -> str:
        lines = [self.answer]
        if self.sources:
            lines.append("\n📚 Sources:")
            lines.extend(f"  • {s}" for s in self.sources)
        if self.tools_used:
            lines.append(f"\n🔧 Tools: {', '.join(sorted(set(self.tools_used)))}")
        if self.processing_time_s:
            lines.append(f"⏱  {self.processing_time_s:.1f}s")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Agent class
# ─────────────────────────────────────────────────────────────────────────────


class MedRAGAgent:
    """
    LangGraph-based ReAct agent for biomedical question answering.

    Uses GPT-4.1-mini by default. Tool calling is driven by the LLM's native
    function-calling API — more reliable than text-parsed ReAct.

    Parameters:
        vector_store:     A loaded :class:`MedRAGVectorStore` instance.
        model:            OpenAI model name.
        temperature:      LLM temperature (keep low for factual tasks).
        max_iterations:   Maximum agent loop iterations before stopping.
        verbose:          Log intermediate messages at DEBUG level.
    """

    def __init__(
        self,
        vector_store: MedRAGVectorStore,
        model: str | None = None,
        temperature: float | None = None,
        max_iterations: int | None = None,
        verbose: bool = False,
    ) -> None:
        from src.config import settings

        self.vector_store = vector_store
        self._fda_client = OpenFDAClient()
        self._verbose = verbose

        _model = model or settings.default_model
        _temp = temperature if temperature is not None else settings.llm_temperature
        self._max_iter = max_iterations or settings.max_agent_iterations

        # LangGraph recursion_limit: each agent→tools round is 2 steps + 1 final
        self._recursion_limit = 2 * self._max_iter + 1

        log.info("Initialising MedRAGAgent | model=%s | max_iter=%d", _model, self._max_iter)

        llm = ChatOpenAI(model=_model, temperature=_temp)
        tools = build_tools(vector_store, self._fda_client)

        self._graph = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
        self._tool_names = [t.name for t in tools]
        log.info("Agent ready | tools=%s", self._tool_names)

    # ─────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────

    def query(self, question: str) -> AgentResponse:
        """
        Answer a biomedical question using RAG + tool calling.

        Args:
            question: Any biomedical question in natural language.

        Returns:
            :class:`AgentResponse` with answer text, citations, and metadata.
        """
        log.info("Query: %r", question[:100])
        t0 = time.perf_counter()

        result = self._graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": self._recursion_limit},
        )

        elapsed = time.perf_counter() - t0
        messages = result.get("messages", [])

        if self._verbose:
            for msg in messages:
                log.debug("  [%s] %s", type(msg).__name__, str(msg.content)[:200])

        answer = self._extract_answer(messages)
        sources, tools_used = self._extract_citations(messages)

        return AgentResponse(
            answer=answer,
            sources=sources,
            tools_used=tools_used,
            processing_time_s=round(elapsed, 2),
            intermediate_steps=messages,
        )

    def stream_query(self, question: str):
        """
        Generator that streams the agent response progressively.

        Yields:
            - ``("status", str)`` tuples each time the agent invokes a tool,
              so the UI can show what the agent is doing in real time.
            - ``("answer", str)`` tuple once the final answer is ready.
            - An :class:`AgentResponse` as the very last item, carrying
              sources, tool names, and timing metadata.

        Example (Streamlit)::

            status_ph = st.empty()
            answer_ph = st.empty()
            response = None
            for item in agent.stream_query(question):
                if isinstance(item, AgentResponse):
                    response = item
                    status_ph.empty()
                elif item[0] == "status":
                    status_ph.markdown(item[1])
                elif item[0] == "answer":
                    answer_ph.markdown(item[1])
        """
        _TOOL_ICONS = {
            "search_pubmed_literature":   "🔍",
            "lookup_fda_drug_info":        "💊",
            "search_drug_in_literature":   "📚",
            "get_adverse_event_statistics": "📊",
        }

        t0 = time.perf_counter()
        all_messages: list = []
        status_lines: list[str] = []

        for chunk in self._graph.stream(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": self._recursion_limit},
            stream_mode="updates",
        ):
            for node_output in chunk.values():
                for msg in node_output.get("messages", []):
                    all_messages.append(msg)

                    if isinstance(msg, AIMessage):
                        # Tool invocations — yield one status line per call
                        if getattr(msg, "tool_calls", None) and not msg.content:
                            for tc in msg.tool_calls:
                                name = (
                                    tc.get("name") if isinstance(tc, dict)
                                    else getattr(tc, "name", "")
                                )
                                args = (
                                    tc.get("args", {}) if isinstance(tc, dict)
                                    else getattr(tc, "args", {})
                                )
                                q_str = (
                                    args.get("query")
                                    or args.get("drug_name")
                                    or ""
                                ).strip()
                                icon = _TOOL_ICONS.get(name, "⚙️")
                                label = (
                                    f"{icon} **{name}**"
                                    + (f': *"{q_str[:55]}"*' if q_str else "")
                                )
                                status_lines.append(label)
                                yield ("status", "\n".join(
                                    f"- {l}" for l in status_lines
                                ))

                        # Final answer text
                        elif msg.content:
                            yield ("answer", str(msg.content))

        elapsed = time.perf_counter() - t0
        answer = self._extract_answer(all_messages)
        sources, tools_used = self._extract_citations(all_messages)

        yield AgentResponse(
            answer=answer,
            sources=sources,
            tools_used=tools_used,
            processing_time_s=round(elapsed, 2),
            intermediate_steps=all_messages,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────

    def _extract_answer(self, messages: list) -> str:
        """Return the content of the last AIMessage that has text."""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return str(msg.content)
        return "No answer generated."

    def _extract_citations(
        self, messages: list
    ) -> tuple[list[str], list[str]]:
        """
        Parse LangGraph messages to extract:
          - citation strings (PMID lines, FDA source lines) from ToolMessages
          - list of tool names called (from AIMessage.tool_calls)
        """
        sources: list[str] = []
        tools_used: list[str] = []
        _citation_markers = ("PMID:", "FDA Drug Label", "NDA ", "Source: FDA", "FAERS")

        for msg in messages:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                    if name:
                        tools_used.append(name)

            elif isinstance(msg, ToolMessage) and isinstance(msg.content, str):
                for line in msg.content.splitlines():
                    stripped = line.strip()
                    if any(marker in stripped for marker in _citation_markers):
                        clean = stripped.lstrip("[]0123456789. •→").strip()
                        if clean and clean not in sources:
                            sources.append(clean)

        return sources[:10], tools_used


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


def build_agent(
    index_path: str | Path | None = None,
    model: str | None = None,
    verbose: bool = False,
) -> MedRAGAgent:
    """
    Load the vector store from disk and return a ready-to-use :class:`MedRAGAgent`.

    This is the main entry point for all consumers (API, CLI, Streamlit).

    Raises:
        RuntimeError: If ``OPENAI_API_KEY`` is not set in the environment.
        RuntimeError: If the FAISS index does not exist yet.
    """
    from src.config import settings

    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n"
            "Add it to your .env file:\n"
            "  OPENAI_API_KEY=sk-...\n\n"
            "Get a key at https://platform.openai.com/api-keys"
        )

    path = Path(index_path) if index_path else settings.vector_store_path
    vs = MedRAGVectorStore(index_path=path)

    if not vs.load():
        raise RuntimeError(
            f"FAISS index not found at: {path}\n\n"
            "Run the ETL pipeline first:\n"
            "  python scripts/run_pipeline.py \\\n"
            "    --query 'your medical topic' \\\n"
            "    --max_results 50 \\\n"
            "    --drugs drug1 drug2\n\n"
            "Or use the ETL panel in the Streamlit sidebar."
        )

    return MedRAGAgent(vector_store=vs, model=model, verbose=verbose)
