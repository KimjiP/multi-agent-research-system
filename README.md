# Multi-Agent Research System (EU AI Act)

A LangGraph research pipeline that turns a question into a cited report. The model says which sources support each claim, and code computes the claim's confidence from those sources. A reviewer checks every report before it is approved, and the reviewer has its own test: errors planted in finished reports. Demoed on EU AI Act compliance; nothing in the architecture is specific to it.

Companion code for the Signal Syntax post [Building and Testing an AI Research Agent in Five Steps: An EU AI Act Example](https://signalsyntax.no/blog/eu-ai-act-research-agent/).

## Three versions

| Version | What it is |
|---|---|
| **V1** | One Claude call, no tools: what the model knows from training |
| **V2** | A fixed LangGraph workflow: researcher, analyst, writer, reviewer |
| **V3** | A supervisor splits the question, researchers search in parallel, no reviewer |

```mermaid
graph LR
    Q([Question]) --> R[Researcher<br/>official EU sites + open web]
    R -->|under 5 sources and 3 rounds| R
    R --> A[Analyst<br/>claims, confidence in code]
    A --> W[Writer]
    W --> RV[Reviewer<br/>link check + 6-point checklist]
    RV -->|a check fails, first round| W
    RV --> E([Approved, or not approved])
```

## Results

Four benchmark questions through all three versions, on Claude Sonnet 5 ([results/](results/)).

- **The deadline that moved.** Asked when the Annex III obligations apply, V1 answers 2 August 2026, the original date. V2 finds the Commission's announcement that the AI Omnibus entered into force on 27 July 2026, and answers 2 December 2027.
- **The reviewer, tested with planted errors** ([eval_reviewer.py](eval_reviewer.py)): 13 of 13 caught (4 invented links, 4 unsupported claims, 4 uncited claims, 1 changed date), and 1 false alarm in 4 clean reports, for $0.72. It checks the report against the extracted claims, so an error already inside a claim passes through.
- **V2 against V3**, for the four questions:

| | V2 | V3 |
|---|---|---|
| Time | 558 s | 381 s |
| Cost | $0.92 | $0.64 |
| Sources per report | 23 to 28 | 59 to 87 |
| Quality gate | Reviewer, 3 of 4 reports approved | None |

## Confidence

`authority × recency × corroboration`, capped at 1.0, computed in [src/confidence.py](src/confidence.py). Authority comes from a fixed list of source types in [src/sources.py](src/sources.py), from official EU text (1.0) down to news and blogs (0.15). Corroboration counts distinct domains. Labels: high from 0.7, medium from 0.4.

## Calibration

Does the confidence score predict whether a claim is correct? The 156 scored claims of the September V2 and V3 runs were checked against official sources (EUR-Lex, the Commission, NANDO and US government pages), with the rules fixed before checking in [calibration/ANALYSIS_PLAN.md](calibration/ANALYSIS_PLAN.md). Claude Opus 5.5 drafted each label with its evidence, blind to the scores, and checked all of them in a second pass. 15 claims could not be settled and are left out.

| Label | Claims checked | Correct | Mean score |
|---|---|---|---|
| High | 63 | 81% | 0.91 |
| Medium | 34 | 85% | 0.62 |
| Low | 44 | 84% | 0.15 |

- **No ranking.** AUROC 0.47 (95% bootstrap interval 0.33 to 0.61), so a correct claim is no more likely than a wrong one to have the higher score.
- **Why.** 22 of the 24 wrong claims were partly right: a date, number or article changed between an official source and the claim. The score rates the source, so it cannot see this, and 12 of the 24 were labeled high.
- **Checking by score.** Sending every claim below 0.7 to a person means checking 58% of all claims, and 12 of the 24 wrong claims still pass.
- **Recalibration** (Platt scaling, 5-fold cross-validation) makes the numbers match only by predicting 78% to 88% for every claim.
- **Other signals.** Asking the agent's model (Claude Sonnet 5) for the probability that each claim is correct ranked claims better: AUROC 0.66 (0.54 to 0.78), calibration error 0.08. Agreement over 5 repeated checks reached 0.61, but the answers differed for only 7 of the 156 claims. The 936 calls cost $2.38.

```bash
uv run python -m calibration.analyze                     # results.json from claims.csv, no API calls
uv run --group analysis python -m calibration.figures    # figures in calibration/figures/
uv run python -m calibration.signals                     # the model's own signals (API, about $2.38)
```

## Run it

Needs Python 3.14+, `uv`, an Anthropic API key and a Tavily API key.

```bash
uv sync
cp .env.example .env   # ANTHROPIC_API_KEY, TAVILY_API_KEY; LangSmith tracing is optional

uv run streamlit run app.py      # UI: benchmark results and live runs
uv run python v1_baseline.py     # V1 benchmark
uv run python run_benchmark.py   # V2 benchmark
uv run python v3_sketch.py       # V3 benchmark
uv run python eval_reviewer.py   # planted-error test of the V2 reviewer
uv run pytest                    # unit tests, no API calls
```

## Limitations

- Few sources have a publication date, so most count as undated.
- Source types come from a hand-made domain list; reputable sites that are not on it count as blogs.
- Four benchmark questions: the results show behavior, not statistics.
