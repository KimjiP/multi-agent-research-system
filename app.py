"""Streamlit UI for the Multi-Agent Research System."""

import json
from pathlib import Path

import streamlit as st

from src.graph import build_graph
from src.state import SourceDoc, Claim, Contradiction
from v3_sketch import run_v3_pipeline

st.set_page_config(page_title="MAS Research Pipeline", layout="wide")
st.title("Multi-Agent Research System")
st.caption("EU AI Act research pipeline — compare V2 (LangGraph) and V3 (parallel multi-agent)")


# =============================================================================
# Shared rendering helpers
# =============================================================================

def _render_metrics(result: dict, version: str = "v2"):
    """Render summary metrics row."""
    sources = result["gathered_sources"]
    claims = result["extracted_claims"]
    contradictions = result["contradictions"]

    if version == "v3":
        cols = st.columns(6)
        cols[0].metric("Sources", len(sources))
        cols[1].metric("Claims", len(claims))
        cols[2].metric("Contradictions", len(contradictions))
        cols[3].metric("Subtopics", len(result.get("subtopics", [])))
        cols[4].metric("Question Type", result.get("question_type", "?"))
        if "duration_seconds" in result:
            cols[5].metric("Duration", f"{result['duration_seconds']:.1f}s")
    else:
        cols = st.columns(5)
        cols[0].metric("Sources", len(sources))
        cols[1].metric("Claims", len(claims))
        cols[2].metric("Contradictions", len(contradictions))
        cols[3].metric("Review Rounds", result.get("review_count", "?"))
        if "duration_seconds" in result:
            cols[4].metric("Duration", f"{result['duration_seconds']:.1f}s")
        else:
            cols[4].metric("Approved", "Yes" if result.get("approved") else "No")


def _render_report(result: dict):
    """Render the report markdown."""
    st.subheader("Research Report")
    st.markdown(result["report"])


def _render_sources(sources, use_pydantic: bool = False):
    """Render expandable source list."""
    st.subheader("Sources")
    for i, s in enumerate(sources, 1):
        title = s.title if use_pydantic else s.get("title", "")
        source_type = s.source_type if use_pydantic else s.get("source_type", "")
        published = (s.published_date if use_pydantic else s.get("published_date")) or "Unknown"
        query = s.search_query if use_pydantic else s.get("search_query", "")
        url = s.url if use_pydantic else s.get("url", "")
        snippet = s.snippet if use_pydantic else s.get("snippet", "")

        with st.expander(f"{i}. {title[:60]}"):
            st.write(f"**Type:** {source_type}")
            st.write(f"**Published:** {published}")
            st.write(f"**Query:** {query}")
            st.write(f"**URL:** {url}")
            st.write(f"**Snippet:** {snippet[:300]}")


def _render_claims(claims, use_pydantic: bool = False):
    """Render claims table."""
    st.subheader("Extracted Claims")
    if not claims:
        return
    claims_data = []
    for c in claims:
        if use_pydantic:
            claims_data.append({
                "ID": c.claim_id,
                "Category": c.category,
                "Confidence": c.confidence,
                "Claim": c.text[:100],
                "Sources": len(c.supporting_sources),
            })
        else:
            claims_data.append({
                "ID": c.get("claim_id", ""),
                "Category": c.get("category", ""),
                "Confidence": c.get("confidence", 0),
                "Claim": c.get("text", "")[:100],
                "Sources": len(c.get("supporting_sources", [])),
            })
    st.dataframe(
        claims_data,
        use_container_width=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="%.2f"
            ),
        },
    )


def _render_issues(contradictions, open_issues, use_pydantic: bool = False):
    """Render contradictions and knowledge gaps."""
    if contradictions:
        st.subheader("Contradictions")
        for ct in contradictions:
            desc = ct.description if use_pydantic else ct.get("description", "")
            st.warning(desc)
    if open_issues:
        st.subheader("Knowledge Gaps")
        for issue in open_issues:
            st.info(issue)


def _render_subtopics(subtopics: list[dict]):
    """Render V3 subtopic decomposition."""
    st.subheader("Subtopic Decomposition")
    for i, st_item in enumerate(subtopics, 1):
        with st.expander(f"Subtopic {i}: {st_item['title']}"):
            for q in st_item["search_queries"]:
                st.write(f"- {q}")


def _render_full_result(result: dict, version: str = "v2", use_pydantic: bool = False):
    """Render a complete pipeline result."""
    _render_metrics(result, version=version)

    if version == "v3" and "subtopics" in result:
        _render_subtopics(result["subtopics"])

    st.divider()
    _render_report(result)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        _render_sources(result["gathered_sources"], use_pydantic=use_pydantic)
    with col2:
        _render_claims(result["extracted_claims"], use_pydantic=use_pydantic)
        _render_issues(
            result["contradictions"],
            result["open_issues"],
            use_pydantic=use_pydantic,
        )


# =============================================================================
# Main layout: V2 and V3 tabs
# =============================================================================

v2_tab, v3_tab = st.tabs(["V2 — LangGraph Pipeline", "V3 — Parallel Multi-Agent"])

# =============================================================================
# V2 Tab
# =============================================================================

V2_BENCHMARK_FILE = Path("v2_benchmark_results_20260331_192522.json")

with v2_tab:
    st.header("V2: LangGraph StateGraph Pipeline")
    st.caption("Researcher → Analyst → Writer → Reviewer with conditional loops")

    # Benchmark results
    if V2_BENCHMARK_FILE.exists():
        with open(V2_BENCHMARK_FILE) as f:
            v2_benchmarks = json.load(f)

        st.subheader("Benchmark Results")
        question_labels = [
            f"Q{i+1}: {r['question'][:65]}..." for i, r in enumerate(v2_benchmarks)
        ]
        benchmark_tabs = st.tabs(question_labels)

        for tab, result in zip(benchmark_tabs, v2_benchmarks):
            with tab:
                st.markdown(f"**{result['question']}**")
                st.caption(
                    f"Run at {result.get('timestamp', 'unknown')} | "
                    f"Queries: {', '.join(result['search_queries'][:3])}"
                )
                _render_full_result(result, version="v2", use_pydantic=False)

        st.divider()

    # Live run
    st.subheader("Run New Query")
    v2_question = st.text_area(
        "Research Question",
        placeholder="Enter a research question to run through the V2 pipeline...",
        height=80,
        key="v2_question",
    )
    v2_run = st.button("Run V2 Pipeline", type="primary", disabled=not v2_question, key="v2_run")

    if v2_run and v2_question:
        graph = build_graph()

        initial_state = {
            "question": v2_question,
            "search_queries": [],
            "gathered_sources": [],
            "iteration_count": 0,
            "research_complete": False,
            "extracted_claims": [],
            "contradictions": [],
            "open_issues": [],
            "report": "",
            "review_feedback": None,
            "approved": False,
            "review_count": 0,
            "current_phase": "",
            "error": None,
        }

        status_container = st.container()
        researcher_status = status_container.status("Researcher", state="running")
        researcher_status.write("Generating search queries and gathering sources...")

        for event in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "researcher":
                    sources = node_output.get("gathered_sources", [])
                    queries = node_output.get("search_queries", [])
                    iteration = node_output.get("iteration_count", 0)
                    researcher_status.write(
                        f"Iteration {iteration}: Found {len(sources)} sources across {len(queries)} queries"
                    )
                    if len(sources) >= 5 or iteration >= 3:
                        researcher_status.update(
                            label=f"Researcher — {len(sources)} sources from {len(queries)} queries",
                            state="complete",
                        )
                        analyst_status = status_container.status("Analyst", state="running")
                        analyst_status.write("Extracting claims and assessing confidence...")

                elif node_name == "analyst":
                    claims = node_output.get("extracted_claims", [])
                    contradictions = node_output.get("contradictions", [])
                    open_issues = node_output.get("open_issues", [])
                    analyst_status.update(
                        label=f"Analyst — {len(claims)} claims, {len(contradictions)} contradictions, {len(open_issues)} open issues",
                        state="complete",
                    )
                    writer_status = status_container.status("Writer", state="running")
                    writer_status.write("Generating research report...")

                elif node_name == "writer":
                    report = node_output.get("report", "")
                    word_count = len(report.split())
                    writer_status.update(
                        label=f"Writer — {word_count} words",
                        state="complete",
                    )
                    reviewer_status = status_container.status("Reviewer", state="running")
                    reviewer_status.write("Evaluating report quality...")

                elif node_name == "reviewer":
                    approved = node_output.get("approved", False)
                    review_count = node_output.get("review_count", 0)
                    feedback = node_output.get("review_feedback")
                    if approved:
                        reviewer_status.update(
                            label=f"Reviewer — Approved (round {review_count})",
                            state="complete",
                        )
                    else:
                        reviewer_status.update(
                            label=f"Reviewer — Revision requested (round {review_count})",
                            state="complete",
                        )
                        writer_status = status_container.status("Writer (revision)", state="running")
                        writer_status.write(f"Revising report: {feedback[:100]}...")

            for key, value in node_output.items():
                initial_state[key] = value

        st.divider()
        _render_full_result(initial_state, version="v2", use_pydantic=True)


# =============================================================================
# V3 Tab
# =============================================================================

V3_BENCHMARK_FILE = Path("v3_sketch_results_20260331_203700.json")

with v3_tab:
    st.header("V3: Parallel Multi-Agent Orchestration")
    st.caption("Supervisor → Parallel Researchers → Analyst → Topic-Aware Writer")

    # Benchmark results
    if V3_BENCHMARK_FILE.exists():
        with open(V3_BENCHMARK_FILE) as f:
            v3_benchmarks = json.load(f)

        st.subheader("Benchmark Results")
        question_labels = [
            f"Q{i+1}: {r['question'][:65]}..." for i, r in enumerate(v3_benchmarks)
        ]
        benchmark_tabs = st.tabs(question_labels)

        for tab, result in zip(benchmark_tabs, v3_benchmarks):
            with tab:
                st.markdown(f"**{result['question']}**")
                st.caption(
                    f"Run at {result.get('timestamp', 'unknown')} | "
                    f"Type: {result.get('question_type', '?')} | "
                    f"Subtopics: {len(result.get('subtopics', []))}"
                )
                _render_full_result(result, version="v3", use_pydantic=False)

        st.divider()

    # Live run
    st.subheader("Run New Query")
    v3_question = st.text_area(
        "Research Question",
        placeholder="Enter a research question to run through the V3 pipeline...",
        height=80,
        key="v3_question",
    )
    v3_run = st.button("Run V3 Pipeline", type="primary", disabled=not v3_question, key="v3_run")

    if v3_run and v3_question:
        status_container = st.container()

        supervisor_status = status_container.status("Supervisor", state="running")
        supervisor_status.write("Decomposing question into subtopics...")

        result = run_v3_pipeline(v3_question)

        supervisor_status.update(
            label=f"Supervisor — {result['question_type']}, {len(result['subtopics'])} subtopics",
            state="complete",
        )
        research_status = status_container.status("Parallel Researchers", state="complete")
        research_status.write(f"Gathered {len(result['gathered_sources'])} sources across {len(result['subtopics'])} subtopics")
        analyst_status = status_container.status("Analyst", state="complete")
        analyst_status.write(f"Extracted {len(result['extracted_claims'])} claims, {len(result['contradictions'])} contradictions")
        writer_status = status_container.status("Writer", state="complete")
        writer_status.write(f"Generated {len(result['report'].split())} word report ({result['question_type']} format)")

        st.divider()
        _render_full_result(result, version="v3", use_pydantic=False)
