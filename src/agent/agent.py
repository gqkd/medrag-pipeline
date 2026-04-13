"""
src/agent/agent.py
───────────────────
MedRAG ReAct agent.

Wraps a LangChain AgentExecutor built on the ReAct pattern:
  Thought → Action → Observation → … → Final Answer

The agent autonomously decides which of its four tools to invoke (and with
what arguments) based solely on the question and the tool descriptions.
Each response includes extracted source citations for traceability.

Build the agent via the module-level factory::

    from src.agent.agent import build_agent
    agent = build_agent()
    response = agent.query("What are the cardiovascular effects of metformin?")
    print(response)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI

from src.agent.tools import build_tools
from src.agent.prompts import build_react_prompt
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
        intermediate_steps:  Raw LangChain step list (action, observation) pairs.
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
    A LangChain ReAct agent specialised for biomedical question answering.

    The agent uses GPT-4.1-mini by default (good balance of cost and quality
    for RAG tasks). Switch to ``gpt-4.1`` for more complex multi-hop questions.

    Parameters:
        vector_store:     A loaded :class:`MedRAGVectorStore` instance.
        model:            OpenAI model name.
        temperature:      LLM temperature (keep low for factual tasks).
        max_iterations:   Maximum ReAct reasoning loops before giving up.
        verbose:          Print agent's reasoning trace to stdout.
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

        _model = model or settings.default_model
        _temp = temperature if temperature is not None else settings.llm_temperature
        _max_iter = max_iterations or settings.max_agent_iterations

        log.info("Initialising MedRAGAgent | model=%s | max_iter=%d", _model, _max_iter)

        llm = ChatOpenAI(model=_model, temperature=_temp)
        tools = build_tools(vector_store, self._fda_client)
        prompt = build_react_prompt()

        react_agent = create_react_agent(llm, tools, prompt)

        self._executor = AgentExecutor(
            agent=react_agent,
            tools=tools,
            verbose=verbose,
            max_iterations=_max_iter,
            handle_parsing_errors=(
                "I encountered a formatting error. Let me try again with a cleaner response."
            ),
            return_intermediate_steps=True,
            early_stopping_method="generate",
        )

        self._tool_names = [t.name for t in tools]
        log.info("Agent ready | tools=%s", self._tool_names)

    # ─────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────

    def query(self, question: str) -> AgentResponse:
        """
        Answer a biomedical question using RAG + tool calling.

        The agent will:
        1. Decide which tools to invoke based on the question
        2. Execute the tools iteratively (ReAct loop)
        3. Synthesise a cited, evidence-based final answer

        Args:
            question: Any biomedical question in natural language.

        Returns:
            :class:`AgentResponse` with answer text, citations, and metadata.

        Raises:
            RuntimeError: If the vector store has not been loaded.
        """
        import time

        log.info("Query: %r", question[:100])
        t0 = time.perf_counter()

        result = self._executor.invoke({
            "input": question,
            "chat_history": "",
        })

        elapsed = time.perf_counter() - t0
        steps = result.get("intermediate_steps", [])
        sources, tools_used = self._extract_citations(steps)

        return AgentResponse(
            answer=result.get("output", "No answer generated."),
            sources=sources,
            tools_used=tools_used,
            processing_time_s=round(elapsed, 2),
            intermediate_steps=steps,
        )

    def stream_query(self, question: str):
        """
        Generator that yields token strings as the agent produces them.
        Useful for streaming UIs (Streamlit, FastAPI StreamingResponse).

        Note: intermediate tool calls are not yielded — only the final
        answer tokens are streamed.
        """
        for event in self._executor.stream({"input": question, "chat_history": ""}):
            if isinstance(event, dict):
                output = event.get("output", "")
                if output:
                    yield output

    # ─────────────────────────────────────────────────────────────
    # Source extraction
    # ─────────────────────────────────────────────────────────────

    def _extract_citations(
        self, steps: list
    ) -> tuple[list[str], list[str]]:
        """
        Parse LangChain intermediate steps to extract:
          - citation strings (PMID lines, FDA source lines)
          - list of tool names that were called
        """
        sources: list[str] = []
        tools_used: list[str] = []

        _citation_markers = ("PMID:", "FDA Drug Label", "NDA ", "Source: FDA", "FAERS")

        for action, observation in steps:
            # Tool name
            if hasattr(action, "tool"):
                tools_used.append(action.tool)

            # Extract citation lines from tool output
            if not isinstance(observation, str):
                continue
            for line in observation.splitlines():
                stripped = line.strip()
                if any(marker in stripped for marker in _citation_markers):
                    # Clean up list prefix like "[1]" or "  •"
                    clean = stripped.lstrip("[]0123456789. •→").strip()
                    if clean and clean not in sources:
                        sources.append(clean)

        return sources[:10], tools_used  # Cap citations at 10


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

    Args:
        index_path: Override the default FAISS index path.
        model:      Override the default LLM model name.
        verbose:    Enable agent reasoning trace output.

    Raises:
        RuntimeError: If the FAISS index does not exist yet.
                      Solution: run ``python scripts/run_pipeline.py --query …``
    """
    from src.config import settings

    path = Path(index_path) if index_path else settings.vector_store_path
    vs = MedRAGVectorStore(index_path=path)

    if not vs.load():
        raise RuntimeError(
            "FAISS index not found at: {path}\n\n"
            "Run the ETL pipeline first:\n"
            "  python scripts/run_pipeline.py \\\n"
            "    --query 'your medical topic' \\\n"
            "    --max_results 50 \\\n"
            "    --drugs drug1 drug2\n\n"
            "Or use the ETL panel in the Streamlit sidebar."
        )

    return MedRAGAgent(vector_store=vs, model=model, verbose=verbose)
