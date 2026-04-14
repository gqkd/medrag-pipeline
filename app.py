"""
app.py
───────
MedRAG Streamlit interface.

Run:
    streamlit run app.py

Requires the FAISS index to be built first:
    python scripts/run_pipeline.py --query "your topic" --drugs drug1 drug2
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Path + env setup ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

# ── Page config — MUST be first Streamlit call ───────────────────────────────
st.set_page_config(
    page_title="MedRAG · Biomedical Research Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark clinical aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0e14;
    color: #c9d1d9;
  }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 2.5rem 3rem; max-width: 1140px; }

  /* ── Typography ── */
  h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; }

  /* ── Header ── */
  .medrag-header {
    display: flex; align-items: flex-end; gap: 18px;
    padding: 24px 0 18px;
    border-bottom: 1px solid #21262d;
    margin-bottom: 28px;
  }
  .medrag-logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.1rem; font-weight: 600;
    color: #58a6ff; letter-spacing: -1px; line-height: 1;
  }
  .medrag-sub {
    font-size: 0.78rem; color: #6e7681; letter-spacing: 1.2px;
    text-transform: uppercase; font-weight: 400;
    padding-bottom: 3px;
  }

  /* ── Stat boxes ── */
  .stat-grid { display: flex; gap: 12px; margin-bottom: 24px; }
  .stat-box {
    flex: 1; background: #161b22;
    border: 1px solid #21262d; border-radius: 8px;
    padding: 14px 18px; text-align: center;
    transition: border-color 0.2s;
  }
  .stat-box:hover { border-color: #30363d; }
  .stat-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.65rem; font-weight: 600; color: #58a6ff;
    line-height: 1.1;
  }
  .stat-label {
    font-size: 0.68rem; color: #6e7681;
    text-transform: uppercase; letter-spacing: 0.9px; margin-top: 5px;
  }

  /* ── Section labels ── */
  .section-label {
    font-size: 0.68rem; color: #6e7681;
    text-transform: uppercase; letter-spacing: 1.1px;
    font-weight: 600; margin: 0 0 8px;
  }

  /* ── Text area ── */
  .stTextArea textarea {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 14px 16px !important;
    caret-color: #58a6ff;
    resize: none;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .stTextArea textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.10) !important;
    outline: none !important;
  }
  .stTextArea textarea::placeholder { color: #484f58 !important; }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 0.88rem !important; padding: 0.55rem 1.6rem !important;
    letter-spacing: 0.2px; cursor: pointer;
    transition: all 0.18s ease;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 5px 18px rgba(31,111,235,0.38) !important;
    filter: brightness(1.1);
  }
  .stButton > button:active { transform: translateY(0); }
  /* Secondary buttons (example chips, clear, etc.) */
  .stButton > button[kind="secondary"] {
    background: #161b22 !important; color: #8b949e !important;
    border: 1px solid #30363d !important;
  }
  .stButton > button[kind="secondary"]:hover {
    border-color: #58a6ff !important; color: #58a6ff !important;
    box-shadow: none !important; transform: none !important;
    filter: none !important;
  }

  /* ── Answer card ── */
  .answer-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #58a6ff;
    border-radius: 10px;
    padding: 22px 26px;
    margin: 20px 0;
    line-height: 1.78; font-size: 0.97rem; color: #e6edf3;
  }
  .answer-card p { margin: 0 0 0.8em; }
  .answer-card p:last-child { margin: 0; }

  /* ── Source chips ── */
  .source-row { margin-top: 12px; }
  .source-chip {
    display: inline-block;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 4px 13px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem; color: #8b949e;
    margin: 3px 3px 3px 0;
    white-space: nowrap; cursor: default;
    transition: all 0.15s;
  }
  .source-chip:hover { color: #c9d1d9; border-color: #58a6ff; }
  .source-chip.pubmed { border-color: rgba(63,185,80,0.4); color: #3fb950; }
  .source-chip.fda    { border-color: rgba(210,153,34,0.4); color: #d29922; }

  /* ── Tool badge ── */
  .tool-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: #0d1117; border: 1px solid #21262d;
    border-radius: 6px; padding: 3px 9px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem; color: #8b949e; margin: 2px;
  }

  /* ── Status dot ── */
  .dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 6px; vertical-align: middle;
  }
  .dot-green  { background: #3fb950; box-shadow: 0 0 5px rgba(63,185,80,0.6); }
  .dot-red    { background: #f85149; }
  .dot-yellow { background: #d29922; animation: blink 1.2s infinite; }
  @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.25; } }

  /* ── Thinking banner ── */
  .thinking-banner {
    background: #161b22; border: 1px solid #21262d;
    border-left: 3px solid #d29922;
    border-radius: 8px; padding: 14px 18px; margin: 16px 0;
    font-size: 0.85rem; color: #8b949e;
  }

  /* ── Not-ready banner ── */
  .not-ready-banner {
    background: #161b22;
    border: 1px solid #30363d; border-left: 3px solid #d29922;
    border-radius: 10px; padding: 20px 24px; margin: 24px 0;
  }
  .not-ready-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem; color: #d29922; font-weight: 600; margin-bottom: 10px;
  }
  .not-ready-body { font-size: 0.88rem; color: #8b949e; line-height: 1.7; }
  code {
    background: #0d1117; padding: 3px 8px;
    border-radius: 4px; color: #58a6ff; font-size: 0.82rem;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* ── Disclaimer ── */
  .disclaimer {
    background: #161b22; border: 1px solid rgba(210,153,34,0.3);
    border-radius: 8px; padding: 9px 15px;
    font-size: 0.76rem; color: #7d6a28; margin-top: 14px;
  }

  /* ── History ── */
  .history-item {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 10px 14px;
    margin-bottom: 7px; font-size: 0.82rem; color: #8b949e;
    cursor: default; transition: all 0.15s;
  }
  .history-item:hover { border-color: #30363d; color: #c9d1d9; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #21262d;
  }
  [data-testid="stSidebar"] .block-container { padding: 1.2rem 1.4rem 2rem; }

  /* ── Select / number inputs ── */
  .stSelectbox > div > div,
  .stNumberInput > div > div > input {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 6px !important;
  }
  .stTextInput > div > div > input {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important; border-radius: 6px !important;
  }

  /* ── Slider ── */
  .stSlider > div > div > div { background: #30363d !important; }
  .stSlider > div > div > div > div { background: #58a6ff !important; }

  /* ── Expander ── */
  details summary {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    color: #8b949e !important;
    font-size: 0.82rem !important;
    padding: 10px 14px !important;
  }

  /* ── Divider ── */
  hr { border-color: #21262d; margin: 16px 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "history": [],          # list[dict] — {question, answer, sources, tools_used, elapsed}
        "current_question": "", # populated by example buttons
        "total_sources": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Cached resources
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_agent(model: str = "gpt-4.1-mini"):
    """Load and cache the agent. Re-runs when `model` changes."""
    try:
        from src.agent.agent import build_agent
        agent = build_agent(model=model, verbose=False)
        return agent, None
    except RuntimeError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"Unexpected error: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding: 6px 0 18px;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.05rem;
                  font-weight:600;color:#58a6ff;letter-spacing:-0.5px;">🧬 MedRAG</div>
      <div style="font-size:0.68rem;color:#6e7681;margin-top:3px;
                  text-transform:uppercase;letter-spacing:1.1px;">
        Biomedical Research Agent
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model selection
    st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "model",
        options=["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"],
        index=0,
        label_visibility="collapsed",
        help="gpt-4.1-mini is fast and cheap. Use gpt-4.1 for harder multi-hop questions.",
    )

    # ── Load agent
    agent, agent_err = _load_agent(model=selected_model)

    # Status
    st.markdown('<div class="section-label" style="margin-top:16px;">Status</div>', unsafe_allow_html=True)
    if agent:
        stats = agent.vector_store.get_stats()
        st.markdown(
            f'<span class="dot dot-green"></span>'
            f'<span style="font-size:0.82rem;color:#3fb950;">Agent online</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span style="font-size:0.75rem;color:#6e7681;">'
            f'📦 {stats.get("total_vectors", 0):,} vectors indexed</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="dot dot-red"></span>'
            '<span style="font-size:0.82rem;color:#f85149;">Agent offline</span>',
            unsafe_allow_html=True,
        )
        if agent_err:
            st.markdown(
                f'<span style="font-size:0.72rem;color:#6e7681;">{agent_err[:120]}</span>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── ETL Pipeline panel
    st.markdown('<div class="section-label">ETL Pipeline</div>', unsafe_allow_html=True)
    with st.expander("⚙️ Ingest new data"):
        etl_query = st.text_input(
            "PubMed query",
            placeholder="e.g. type 2 diabetes treatment",
            key="etl_query",
        )
        etl_max = st.number_input(
            "Max articles", min_value=5, max_value=200, value=30, step=5
        )
        etl_drugs_raw = st.text_input(
            "Drugs (comma-separated)",
            placeholder="metformin, semaglutide",
            key="etl_drugs",
        )
        etl_append = st.checkbox("Append to existing index", value=False)

        if st.button("▶ Run Pipeline", use_container_width=True):
            if not etl_query.strip():
                st.warning("Enter a PubMed query first.")
            else:
                drug_list = [
                    d.strip() for d in etl_drugs_raw.split(",") if d.strip()
                ]
                with st.spinner("Running ETL pipeline... this may take a few minutes."):
                    try:
                        from scripts.run_pipeline import run_pipeline
                        run_stats = run_pipeline(
                            query=etl_query.strip(),
                            max_results=int(etl_max),
                            drug_names=drug_list,
                            append=etl_append,
                            quiet=True,
                        )
                        st.success(
                            f"✅ Done! "
                            f"{run_stats['articles_fetched']} articles · "
                            f"{run_stats['chunks_indexed']} chunks indexed."
                        )
                        st.cache_resource.clear()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Pipeline error: {exc}")

    st.markdown("---")

    # ── Query history
    if st.session_state.history:
        st.markdown('<div class="section-label">Recent</div>', unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-7:]):
            preview = item["question"]
            if len(preview) > 55:
                preview = preview[:55] + "…"
            st.markdown(
                f'<div class="history-item">💬 {preview}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑 Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.total_sources = 0
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="medrag-header">
  <div>
    <div class="medrag-logo">MedRAG Pipeline</div>
    <div class="medrag-sub">Evidence-based answers · PubMed + FDA · Cited sources</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Stats row ─────────────────────────────────────────────────────────────────
if agent:
    vs_stats = agent.vector_store.get_stats()
    c1, c2, c3, c4 = st.columns(4)
    for col, value, label in [
        (c1, f"{vs_stats.get('total_vectors', 0):,}", "Vectors Indexed"),
        (c2, "4", "Tools Available"),
        (c3, str(len(st.session_state.history)), "Queries Run"),
        (c4, str(st.session_state.total_sources), "Sources Cited"),
    ]:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div class="stat-value">{value}</div>'
                f'<div class="stat-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)

# ── Not-ready banner ──────────────────────────────────────────────────────────
if not agent:
    st.markdown(f"""
    <div class="not-ready-banner">
      <div class="not-ready-title">⚠️ Vector Index Not Found</div>
      <div class="not-ready-body">
        The FAISS vector index hasn't been built yet. Run the ETL pipeline
        to ingest PubMed articles and FDA drug data into the search index.<br><br>
        <strong>From the terminal:</strong><br>
        <code>python scripts/run_pipeline.py --query "diabetes" --drugs metformin semaglutide</code><br><br>
        Or use the <strong>ETL Pipeline</strong> panel in the sidebar.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Query input area ──────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Ask a Biomedical Question</div>', unsafe_allow_html=True)

_EXAMPLES = [
    "What are the latest treatments for Type 2 Diabetes?",
    "What are the drug interactions of warfarin?",
    "Compare GLP-1 agonists vs SGLT-2 inhibitors for cardiovascular outcomes",
    "What does the literature say about metformin and cancer prevention?",
    "What are the FDA warnings for semaglutide (Ozempic)?",
    "What are the mechanisms of action of ACE inhibitors?",
]

col_q, col_ex = st.columns([3, 1])

with col_q:
    # Pre-fill from example buttons via session state
    default_val = st.session_state.get("current_question", "")
    question = st.text_area(
        "question",
        value=default_val,
        placeholder=(
            "e.g. What are the side effects of metformin and what does "
            "recent literature say about its long-term safety?"
        ),
        height=110,
        label_visibility="collapsed",
        key="question_input",
    )

with col_ex:
    st.markdown(
        '<div style="font-size:0.68rem;color:#6e7681;margin-bottom:6px;">'
        'Quick examples</div>',
        unsafe_allow_html=True,
    )
    for ex in _EXAMPLES[:5]:
        if st.button(ex[:42] + "…", key=f"ex_{ex[:20]}", use_container_width=True):
            st.session_state["current_question"] = ex
            st.rerun()

# ── Submit row
btn_col, hint_col = st.columns([1, 4])
with btn_col:
    run_query = st.button(
        "🔬  Search & Analyze",
        use_container_width=True,
        type="primary",
        disabled=not agent,
    )
with hint_col:
    st.markdown(
        '<div style="font-size:0.74rem;color:#484f58;padding-top:11px;">'
        'The agent searches PubMed literature and FDA data, then returns a cited answer.'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Run query
# ─────────────────────────────────────────────────────────────────────────────

if run_query:
    q = question.strip()
    if not q:
        st.warning("Please type a question before clicking Search.")
    elif not agent:
        st.error("Build the vector index first (see sidebar).")
    else:
        # Clear pre-filled example
        st.session_state["current_question"] = ""

        # Streaming: show tool status live, then the answer
        status_ph = st.empty()
        answer_ph = st.empty()
        response = None

        status_ph.markdown(
            '<div class="thinking-banner">'
            '<span class="dot dot-yellow"></span>'
            'Agent reasoning — searching literature and FDA data…'
            '</div>',
            unsafe_allow_html=True,
        )

        from src.agent.agent import AgentResponse
        try:
            for item in agent.stream_query(q):
                if isinstance(item, AgentResponse):
                    response = item
                    status_ph.empty()
                elif item[0] == "status":
                    status_ph.markdown(
                        '<div class="thinking-banner">'
                        '<span class="dot dot-yellow"></span>'
                        f'{item[1]}'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                elif item[0] == "answer":
                    answer_ph.markdown(item[1] + " ▌")
        except Exception as exc:
            status_ph.empty()
            answer_ph.empty()
            st.error(f"Agent error: {exc}")
            st.stop()

        answer_ph.empty()

        if response is None:
            st.error("No response received from the agent.")
            st.stop()

        elapsed = response.processing_time_s

        # Persist to history
        st.session_state.history.append({
            "question": q,
            "answer": response.answer,
            "sources": response.sources,
            "tools_used": response.tools_used,
            "elapsed": elapsed,
        })
        st.session_state.total_sources += len(response.sources)

        # ── Answer card
        # Convert plain newlines to <br> for rendering inside HTML
        answer_html = response.answer.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        answer_html = answer_html.replace("\n\n", "</p><p>").replace("\n", "<br>")
        st.markdown(
            f'<div class="answer-card"><p>{answer_html}</p></div>',
            unsafe_allow_html=True,
        )

        # ── Meta: time + tools
        mc1, mc2 = st.columns([1, 4])
        with mc1:
            st.markdown(
                f'<div style="font-size:0.74rem;color:#6e7681;">⏱ {elapsed:.1f}s</div>',
                unsafe_allow_html=True,
            )
        with mc2:
            badges = "".join(
                f'<span class="tool-badge">⚙ {t}</span>'
                for t in sorted(set(response.tools_used))
            )
            st.markdown(badges, unsafe_allow_html=True)

        # ── Source chips
        if response.sources:
            st.markdown(
                '<div class="section-label" style="margin-top:18px;">Sources</div>',
                unsafe_allow_html=True,
            )
            chips = ""
            for src in response.sources:
                css_cls = "pubmed" if "PMID" in src else "fda" if ("FDA" in src or "NDA" in src) else ""
                icon = "📄" if "PMID" in src else "💊" if css_cls == "fda" else "🔗"
                short = src[:90] + ("…" if len(src) > 90 else "")
                chips += f'<span class="source-chip {css_cls}">{icon} {short}</span>'
            st.markdown(
                f'<div class="source-row">{chips}</div>',
                unsafe_allow_html=True,
            )

        # ── Disclaimer
        st.markdown(
            '<div class="disclaimer">'
            '⚕️ <strong>Research use only.</strong> '
            'This tool synthesises publicly available scientific literature and FDA data. '
            'Always consult a qualified healthcare professional for medical decisions.'
            '</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Previous results accordion
# ─────────────────────────────────────────────────────────────────────────────

prev = [
    h for h in st.session_state.history
    if h["question"] != question.strip()
]
if prev:
    st.markdown("---")
    st.markdown(
        '<div class="section-label">Previous Queries in This Session</div>',
        unsafe_allow_html=True,
    )
    for item in reversed(prev[-6:]):
        label = item["question"][:92] + ("…" if len(item["question"]) > 92 else "")
        with st.expander(f"💬 {label}"):
            # Answer
            ans_html = item["answer"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            ans_html = ans_html.replace("\n\n", "</p><p>").replace("\n", "<br>")
            st.markdown(
                f'<div class="answer-card"><p>{ans_html}</p></div>',
                unsafe_allow_html=True,
            )
            # Sources
            if item.get("sources"):
                chips = "".join(
                    f'<span class="source-chip {"pubmed" if "PMID" in s else "fda" if "FDA" in s or "NDA" in s else ""}">'
                    f'{"📄" if "PMID" in s else "💊"} {s[:70]}</span>'
                    for s in item["sources"]
                )
                st.markdown(f'<div class="source-row">{chips}</div>', unsafe_allow_html=True)
            # Footer
            tools_str = ", ".join(sorted(set(item.get("tools_used", []))))
            st.markdown(
                f'<div style="font-size:0.71rem;color:#6e7681;margin-top:10px;">'
                f'⏱ {item["elapsed"]:.1f}s  ·  🔧 {tools_str or "—"}</div>',
                unsafe_allow_html=True,
            )
