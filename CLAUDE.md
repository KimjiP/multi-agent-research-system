# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-Agent Research System (MAS) — a LangGraph-based pipeline that takes a research question and produces a structured, cited report. Demoed with EU AI Act compliance as the vertical but architecturally generic.

Three versions at increasing complexity:
- **V1** (`v1_baseline.py`): Single Claude API call baseline
- **V2** (`src/`): LangGraph StateGraph with 4 nodes — the main deliverable
- **V3** (`v3_sketch.py`): Multi-agent orchestration demo; reuses V2's search, Analyst and confidence scoring

## Commands

```bash
uv sync                          # Install dependencies
uv run streamlit run app.py      # Streamlit UI (V1, V2, V3 benchmarks and live queries)
uv run python v1_baseline.py     # Run V1 baseline (saves results/v1_*.json)
uv run python run_benchmark.py   # Run V2 benchmarks (saves results/v2_*.json)
uv run python v3_sketch.py       # Run V3 benchmarks (saves results/v3_*.json)
uv run python eval_reviewer.py   # Planted-error test of the V2 Reviewer
uv run pytest                    # Unit tests (no API calls)
```

## Architecture (V2)

The V2 pipeline is a LangGraph `StateGraph` over a shared `ResearchState` TypedDict. All nodes read/write to this single state object.

**Graph flow:** `Researcher → [loop?] → Analyst → Writer → Reviewer → [revise?] → END`

- **Researcher** (`src/nodes/researcher.py`) — Tool-calling node. Generates search queries via LLM, runs them via Tavily (one of them restricted to `OFFICIAL_DOMAINS`), classifies each source deterministically (`src/sources.py`). Loops up to `MAX_SEARCH_ITERATIONS` or until `MIN_SOURCES` reached.
- **Analyst** (`src/nodes/analyst.py`) — Extracts claims, contradictions and open issues with a structured output. Claims cite sources by number; claims with no valid source are dropped. Confidence is computed in code (`src/confidence.py`), never by the LLM.
- **Writer** (`src/nodes/writer.py`) — Produces the markdown report from analyst output. On revision, incorporates `review_feedback` rather than regenerating.
- **Reviewer** (`src/nodes/reviewer.py`) — A deterministic link check plus a six-item LLM checklist. Approves only when every check passes; if the review itself fails, the report is not approved (fail closed). At most `MAX_REVIEWS` rounds.

**Key files:**
- `src/state.py` — `ResearchState` TypedDict + Pydantic models (`SourceDoc`, `Claim`, `Confidence`, `Contradiction`, `CheckResult`)
- `src/config.py` — All thresholds, weights, effort levels and domain mappings. Loads `.env` via python-dotenv.
- `src/llm.py` — The only place Claude is called: `generate_text` and `generate_structured` (Anthropic SDK, `messages.parse` with Pydantic models). Tracks token usage for cost reporting.
- `src/edges.py` — Deterministic conditional edge functions (`should_continue_research`, `review_decision`)
- `src/graph.py` — StateGraph assembly and compilation
- `src/benchmarks.py` — The benchmark questions shared by V1, V2 and V3, and the Q4 currency check

## Environment Variables

Copy `.env.example` to `.env`. Required keys: `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`. Optional: `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_ENDPOINT`, `LANGCHAIN_PROJECT` for LangSmith tracing. Use `LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com` for EU instance.

## Design Decisions

- `claude-sonnet-5` for all LLM calls (not Opus — cost/quality balance). Sonnet 5 rejects non-default `temperature`, so none is set; `effort` is set per node in `config.EFFORT`.
- The Anthropic SDK is called directly. LangSmith's `wrap_anthropic` does not support anthropic 1.x, so `src/llm.py` traces calls with `@traceable`.
- No facts are hardcoded in prompts. The prompts carry today's date; current information has to come from the sources. (The March 2026 version hardcoded "Digital Omnibus: proposed, not adopted", which became false in July 2026.)
- Source authority hierarchy in `config.py` (`SOURCE_AUTHORITY_WEIGHTS`) drives confidence scoring
- The PRD with full specifications is in `llm/PRD.md`
