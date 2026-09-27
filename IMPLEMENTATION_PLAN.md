# Implementation Plan

Step-by-step build order for the Multi-Agent Research System. Each step produces something testable before moving on.

---

## Step 1: Project scaffolding

- Create `src/__init__.py`, `src/nodes/__init__.py`
- Create `src/state.py` — `ResearchState` TypedDict + Pydantic models (`SourceDoc`, `Claim`, `Contradiction`)
- Create `src/config.py` — all constants, weights, domain mappings, loads `.env` via python-dotenv
- Create `.env` from `.env.example` with real API keys
- **Verify:** import `src.state` and `src.config` without errors

## Step 2: Researcher node

- Create `src/nodes/researcher.py`
- Implement `_classify_source_type(url)` — domain-based heuristic using `DOMAIN_TO_SOURCE_TYPE` from config, with fallbacks for `.gov`/`.europa.eu` domains
- Implement `_generate_queries(question, iteration, previous_queries, sources)` — uses ChatAnthropic to generate `QUERIES_PER_ITERATION` search queries; initial iteration from question, follow-ups target gaps
- Implement `researcher_node(state) -> dict` — generates queries, executes Tavily searches (`search_depth="advanced"`, `max_results=5`), creates `SourceDoc`s, deduplicates by URL, increments `iteration_count`
- **Verify:** run researcher node standalone with a test question, confirm it returns populated `gathered_sources`

## Step 3: Analyst node

- Create `src/nodes/analyst.py`
- Implement `analyst_node(state) -> dict`
- System prompt must include: source authority weights, recency multipliers, confidence formula (`authority × recency × corroboration`), domain-specific alerts (EO 14110 revoked, Digital Omnibus proposed, GPAI/high-risk timelines)
- Format all `gathered_sources` into the prompt with url, title, snippet, source_type, published_date
- Instruct LLM to return JSON-only output matching `Claim`, `Contradiction`, and `open_issues` schemas
- Parse LLM JSON response into Pydantic models; handle malformed JSON gracefully
- **Verify:** feed researcher output into analyst, confirm structured claims with confidence scores are produced

## Step 4: Writer node

- Create `src/nodes/writer.py`
- Implement `writer_node(state) -> dict`
- System prompt defines report structure: Executive Summary, Key Findings (by category, with citations and confidence), Contradictions & Uncertainties, Knowledge Gaps, Sources list, Methodology Note
- Two modes: initial generation (from claims/contradictions/open_issues) and revision (incorporates `review_feedback` against existing `report`)
- **Verify:** feed analyst output into writer, confirm markdown report with citations is produced

## Step 5: Reviewer node

- Create `src/nodes/reviewer.py`
- Implement `reviewer_node(state) -> dict`
- System prompt includes 6-point checklist: citation coverage, hallucination check (claims must trace to `extracted_claims`), contradiction disclosure, confidence assessment, executive summary accuracy, knowledge gaps
- Returns `approved=True` + `review_feedback=None` if all checks pass, or `approved=False` + specific revision instructions
- Always increments `review_count`
- **Verify:** feed writer output into reviewer, confirm it produces approval or actionable feedback

## Step 6: Edges and graph assembly

- Create `src/edges.py` — `should_continue_research(state)` and `review_decision(state)`, both deterministic based on config thresholds
- Create `src/graph.py` — `build_graph()` assembles `StateGraph(ResearchState)` with all 4 nodes, conditional edges for researcher loop and reviewer loop, compiles with `checkpointer=None`
- **Verify:** run full graph end-to-end with one benchmark question, confirm it completes all phases

## Step 7: End-to-end testing with benchmark questions

- Run all 3 benchmark questions through the V2 pipeline:
  1. High-risk AI hiring tool compliance requirements
  2. EU AI Act vs US federal AI policy (must detect EO 14110 revocation)
  3. Conformity assessment procedures + notified body status
- Save outputs as JSON for comparison
- Verify: 5+ cited sources per report, no hallucinated references, EO 14110 flagged as revoked in Q2

## Step 8: Fix v1_baseline.py for local execution

- Remove Colab-specific code (`!pip install`, `from google.colab import userdata`)
- Load API key via `python-dotenv` instead
- Ensure it runs locally with `uv run python v1_baseline.py`

## Step 9: Streamlit UI

- Create `app.py`
- Input: text box for research question + "Run" button
- Pipeline status: show `current_phase` with `st.status` as each node completes
- Progress panel: per-node summaries (source count, claim count, word count, approval status)
- Report output: render final `report` as markdown
- Source explorer: expandable section with all `gathered_sources` metadata
- Claims table: sortable table of `extracted_claims` with confidence scores
- Run graph with streaming/callbacks to update UI in real-time

## Step 10: LangSmith observability

- Configure env vars for LangSmith tracing (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`)
- Add `logging.getLogger(__name__)` calls in each node with phase transitions and key metrics
- Run a benchmark question and verify the full trace appears in LangSmith dashboard
- Capture a screenshot of the trace for the README

## Step 11: V3 sketch

- Create `v3_sketch.py`
- Supervisor node: decomposes question into subtopics via LLM
- Parallel researchers: one per subtopic, run concurrently
- Topic-aware writer: different prompts based on topic classification
- Run same benchmark questions; capture cost/latency metrics from LangSmith for comparison table

## Step 12: README and final polish

- Write README following the PRD structure: title, three versions narrative, architecture diagram (mermaid), how it works walkthrough, confidence-weighted synthesis explanation, sample output, V1 vs V2 comparison (EO 14110 example), V3 tradeoffs table, LangSmith trace screenshot, setup instructions, tech stack + design decisions
- Clean up any dead code or unused files
- Final end-to-end run of all 3 versions on all 3 benchmark questions
