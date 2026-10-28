"""Streamlit UI for the Multi-Agent Research System.

Live runs call the APIs and cost money. On Google Cloud Run the app is read-only:
it shows the saved results, and live runs need the access code in LIVE_RUN_CODE.
"""

import hmac
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path

import streamlit as st

from run_benchmark import serialize_state
from src.benchmarks import mentions_current_annex_iii_date
from src.graph import build_graph
from src.llm import usage
from src.state import initial_state
from v1_baseline import run_v1_baseline
from v3_sketch import run_v3_pipeline

RESULTS_DIR = Path("results")

st.set_page_config(page_title="MAS Research Pipeline", layout="wide")
st.title("Multi-Agent Research System")
st.caption(
    "EU AI Act research at three levels of complexity: V1 (one LLM call), "
    "V2 (LangGraph pipeline with a review loop), V3 (parallel multi-agent sketch)"
)

STATUS_LABELS = {
    "approved": "Approved",
    "not_approved": "Not approved after the last review round",
    "review_failed": "Review failed, not approved",
    "revision_requested": "Revision requested",
    "pending": "Not reviewed",
}


def live_runs_allowed() -> bool:
    """Whether this visitor may start live runs.

    Without LIVE_RUN_CODE they are allowed on your own machine, as before, and never
    on Cloud Run (which sets K_SERVICE), so a missing secret leaves the app read-only.
    """
    code = os.getenv("LIVE_RUN_CODE")
    if not code:
        if os.getenv("K_SERVICE"):
            st.sidebar.info("Live runs are turned off. The tabs show saved benchmark results.")
            return False
        return True
    entered = st.sidebar.text_input("Access code for live runs", type="password")
    if hmac.compare_digest(entered.encode(), code.encode()):
        return True
    if entered:
        st.sidebar.error("Wrong access code.")
    st.sidebar.info("The tabs show saved benchmark results. Live runs need an access code.")
    return False


@st.cache_resource
def _live_run_lock() -> threading.Lock:
    """One lock shared by every visitor."""
    return threading.Lock()


def run_live(run) -> None:
    """Call run() unless another live run is going on, so there is one live run at a time."""
    lock = _live_run_lock()
    if not lock.acquire(blocking=False):
        st.warning("Another live run is going on. Please try again in a few minutes.")
        return
    try:
        run()
    finally:
        lock.release()


def latest_results(prefix: str) -> tuple[Path | None, list[dict]]:
    files = sorted(RESULTS_DIR.glob(f"{prefix}_*.json"))
    if not files:
        return None, []
    with open(files[-1]) as f:
        return files[-1], json.load(f)


# =============================================================================
# Rendering helpers (results are plain dicts, as saved by the benchmark scripts)
# =============================================================================


def _render_currency_check(result: dict, text: str) -> None:
    if "Annex III" in result["question"] and "start to apply" in result["question"]:
        if mentions_current_annex_iii_date(text):
            st.success("Gives the current date: 2 December 2027, set by the 2026 Digital Omnibus.")
        else:
            st.error("Does not give the current date (2 December 2027, set by the 2026 Digital Omnibus).")


def _render_metrics(result: dict, version: str) -> None:
    cols = st.columns(6)
    cols[0].metric("Sources", len(result["gathered_sources"]))
    cols[1].metric("Claims", len(result["extracted_claims"]))
    cols[2].metric("Contradictions", len(result["contradictions"]))
    if version == "v2":
        cols[3].metric("Review rounds", result["review_count"])
    else:
        cols[3].metric("Subtopics", len(result.get("subtopics", [])))
    cols[4].metric("Duration", f"{result['duration_seconds']:.0f}s")
    cols[5].metric("Cost", f"${result['cost_usd']:.3f}")
    if version == "v2":
        status = result["review_status"]
        message = f"Reviewer: {STATUS_LABELS.get(status, status)}"
        (st.success if status == "approved" else st.warning)(message)


def _render_checklist(result: dict) -> None:
    checklist = result.get("review_checklist") or {}
    if not checklist:
        return
    st.subheader("Reviewer checklist (last round)")
    for name, check in checklist.items():
        st.markdown(f"{'✅' if check['passed'] else '❌'} **{name}**: {check['notes']}")


def _render_sources(sources: list[dict]) -> None:
    st.subheader("Sources")
    types = Counter(s["source_type"] for s in sources)
    dated = sum(1 for s in sources if s.get("published_date"))
    st.caption(
        ", ".join(f"{t}: {n}" for t, n in types.most_common()) + f" | {dated} of {len(sources)} dated"
    )
    for i, s in enumerate(sources, 1):
        with st.expander(f"{i}. {s['title'][:70]}"):
            st.write(f"**Type:** {s['source_type']}  |  **Published:** {s.get('published_date') or 'no date'}")
            st.write(f"**Query:** {s['search_query']}")
            st.write(f"**URL:** {s['url']}")
            st.write(f"**Snippet:** {s['snippet'][:300]}")


def _render_claims(claims: list[dict]) -> None:
    st.subheader("Extracted claims")
    st.caption(
        "Confidence = authority × recency × corroboration, computed in code from the "
        "supporting sources. Undated sources get recency 0.8."
    )
    if not claims:
        return
    rows = []
    for c in claims:
        conf = c["confidence"]
        rows.append(
            {
                "ID": c["claim_id"],
                "Confidence": conf["score"],
                "Authority": conf["authority"],
                "Recency": f"{conf['recency']}{' (undated)' if conf['undated'] else ''}",
                "Corroboration": conf["corroboration"],
                "Sources": len(c["supporting_sources"]),
                "Category": c["category"],
                "Claim": c["text"],
            }
        )
    st.dataframe(
        rows,
        use_container_width=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
        },
    )


def _render_issues(result: dict) -> None:
    if result["contradictions"]:
        st.subheader("Contradictions")
        for ct in result["contradictions"]:
            st.warning(ct["description"])
    if result["open_issues"]:
        st.subheader("Knowledge gaps")
        for issue in result["open_issues"]:
            st.info(issue)


def render_pipeline_result(result: dict, version: str) -> None:
    _render_metrics(result, version)
    _render_currency_check(result, result["report"])
    if version == "v3" and result.get("subtopics"):
        with st.expander(f"Subtopics ({result.get('question_type', '?')})"):
            for sub in result["subtopics"]:
                st.markdown(f"**{sub['title']}**: " + "; ".join(sub["search_queries"]))
    st.divider()
    st.subheader("Research report")
    st.markdown(result["report"])
    st.divider()
    if version == "v2":
        _render_checklist(result)
    col1, col2 = st.columns([1, 2])
    with col1:
        _render_sources(result["gathered_sources"])
    with col2:
        _render_claims(result["extracted_claims"])
        _render_issues(result)


def render_benchmarks(prefix: str, render) -> None:
    path, results = latest_results(prefix)
    if not results:
        st.info(f"No saved benchmark results in {RESULTS_DIR}/ yet.")
        return
    st.subheader("Benchmark results")
    st.caption(f"{path.name}, model {results[0].get('model', '?')}")
    tabs = st.tabs([f"Q{i + 1}" for i in range(len(results))])
    for tab, result in zip(tabs, results):
        with tab:
            st.markdown(f"**{result['question']}**")
            st.caption(f"Run at {result.get('timestamp', 'unknown')}")
            render(result)


def render_v1(result: dict) -> None:
    cols = st.columns(3)
    cols[0].metric("Duration", f"{result['duration_seconds']:.0f}s")
    cols[1].metric("Output tokens", result["output_tokens"])
    cols[2].metric("Cost", f"${result['cost_usd']:.3f}")
    _render_currency_check(result, result["response"])
    st.markdown(result["response"])


def live_v1(question: str) -> None:
    with st.spinner("Calling the model ..."):
        result = run_v1_baseline(question)
    render_v1(result)


def live_v2(question: str) -> None:
    graph = build_graph()
    state = initial_state(question)
    usage.reset()
    start = time.time()
    status = st.status("Researcher: searching ...", state="running")
    for event in graph.stream(state, stream_mode="updates"):
        for node_name, output in event.items():
            state.update(output)
            if node_name == "researcher":
                status.write(f"Researcher: {len(state['gathered_sources'])} sources so far")
            elif node_name == "analyst":
                status.write(
                    f"Analyst: {len(state['extracted_claims'])} claims, "
                    f"{len(state['contradictions'])} contradictions"
                )
            elif node_name == "writer":
                status.write(f"Writer: {len(state['report'].split())} words")
            elif node_name == "reviewer":
                status.write(
                    f"Reviewer round {state['review_count']}: "
                    f"{STATUS_LABELS.get(state['review_status'], state['review_status'])}"
                )
    status.update(label="Done", state="complete")
    result = serialize_state(state)
    result.update(duration_seconds=time.time() - start, cost_usd=usage.cost_usd)
    st.divider()
    render_pipeline_result(result, "v2")


def live_v3(question: str) -> None:
    with st.spinner("Supervisor, parallel research, analysis and writing ..."):
        result = run_v3_pipeline(question)
    render_pipeline_result(result, "v3")


# =============================================================================
# Tabs
# =============================================================================

LIVE_RUNS = live_runs_allowed()

v1_tab, v2_tab, v3_tab = st.tabs(
    ["V1 — Single LLM call", "V2 — LangGraph pipeline", "V3 — Parallel multi-agent"]
)

with v1_tab:
    st.header("V1: one LLM call, no tools")
    st.caption("The baseline: what the model says from its training data alone, with no sources.")
    render_benchmarks("v1", render_v1)
    if LIVE_RUNS:
        st.divider()
        st.subheader("Run new query")
        v1_question = st.text_area("Research question", key="v1_question", height=80)
        if st.button("Run V1", type="primary", disabled=not v1_question, key="v1_run"):
            run_live(lambda: live_v1(v1_question))

with v2_tab:
    st.header("V2: LangGraph StateGraph pipeline")
    st.caption("Researcher → Analyst → Writer → Reviewer, with a research loop and a review loop")
    render_benchmarks("v2", lambda r: render_pipeline_result(r, "v2"))

    if LIVE_RUNS:
        st.divider()
        st.subheader("Run new query")
        v2_question = st.text_area("Research question", key="v2_question", height=80)
        if st.button("Run V2 pipeline", type="primary", disabled=not v2_question, key="v2_run"):
            run_live(lambda: live_v2(v2_question))

with v3_tab:
    st.header("V3: supervisor and parallel researchers")
    st.caption("Supervisor → parallel researchers → Analyst → topic-aware Writer, no review loop")
    render_benchmarks("v3", lambda r: render_pipeline_result(r, "v3"))

    if LIVE_RUNS:
        st.divider()
        st.subheader("Run new query")
        v3_question = st.text_area("Research question", key="v3_question", height=80)
        if st.button("Run V3 pipeline", type="primary", disabled=not v3_question, key="v3_run"):
            run_live(lambda: live_v3(v3_question))
