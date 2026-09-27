# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-Agent Research System (MAS) — a LangGraph-based pipeline that takes a research question and produces a structured, cited report. Demoed with EU AI Act compliance as the vertical but architecturally generic.

Three versions at increasing complexity:
- **V1** (`v1_baseline.py`): Single Claude API call baseline
- **V2** (`src/`): LangGraph StateGraph with 4 nodes — the main deliverable
- **V3** (`v3_sketch.py`): Multi-agent orchestration demo (build last)

## Commands

```bash
uv sync                          # Install dependencies
uv run streamlit run app.py      # Run the Streamlit UI (V2 + V3 benchmarks and live queries)
uv run python v1_baseline.py     # Run V1 baseline
uv run python run_benchmark.py   # Run V2 benchmarks (saves JSON results)
uv run python v3_sketch.py       # Run V3 benchmarks (saves JSON results)
```

## Architecture (V2)

The V2 pipeline is a LangGraph `StateGraph` over a shared `ResearchState` TypedDict. All nodes read/write to this single state object.

**Graph flow:** `Researcher → [loop?] → Analyst → Writer → Reviewer → [revise?] → END`

- **Researcher** (`src/nodes/researcher.py`) — L3 tool-calling node. Generates search queries via LLM, executes them via Tavily API, collects `SourceDoc`s. Only node that makes external tool calls. Loops up to `MAX_SEARCH_ITERATIONS` or until `MIN_SOURCES` reached.
- **Analyst** (`src/nodes/analyst.py`) — Pure LLM reasoning. Extracts `Claim`s from sources, computes confidence scores (`authority × recency × corroboration`), detects `Contradiction`s.
- **Writer** (`src/nodes/writer.py`) — Produces the formatted markdown report from analyst output. On revision, incorporates `review_feedback` rather than regenerating.
- **Reviewer** (`src/nodes/reviewer.py`) — Evaluates report against extracted claims. Approves or sends revision feedback. Max `MAX_REVIEWS` loops.

**Key files:**
- `src/state.py` — `ResearchState` TypedDict + Pydantic models (`SourceDoc`, `Claim`, `Contradiction`)
- `src/config.py` — All thresholds, weights, and domain mappings. Loads `.env` via python-dotenv.
- `src/edges.py` — Deterministic conditional edge functions (`should_continue_research`, `review_decision`)
- `src/graph.py` — StateGraph assembly and compilation

## Environment Variables

Copy `.env.example` to `.env`. Required keys: `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`. Optional: `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_ENDPOINT`, `LANGCHAIN_PROJECT` for LangSmith tracing. Use `LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com` for EU instance.

## Design Decisions

- Use `claude-sonnet-4-20250514` for all LLM calls (not Opus — cost/quality balance)
- Source authority hierarchy in `config.py` (`SOURCE_AUTHORITY_WEIGHTS`) drives confidence scoring
- Domain-specific recency rules: US EO 14110 was revoked Jan 2025, EU Digital Omnibus is proposed not adopted, GPAI obligations effective Aug 2025, high-risk obligations from Aug 2026
- The PRD with full specifications is in `llm/PRD.md`
