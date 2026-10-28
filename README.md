# Multi-Agent Research System (EU AI Act)

[![Tests](https://github.com/KimjiP/multi-agent-research-system/actions/workflows/tests.yml/badge.svg)](https://github.com/KimjiP/multi-agent-research-system/actions/workflows/tests.yml)

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
- **The reviewer, tested with planted errors** ([eval_reviewer.py](eval_reviewer.py)): 13 errors planted in the reports of a saved run (4 invented links, 4 unsupported claims, 4 uncited claims, 1 changed date). In 6 runs on two saved runs it caught 75 of the 78, and the 3 it missed were uncited claims. It also rejected 1 to 3 of 4 unchanged reports, at about $0.80 a run. It checks the report against the extracted claims, so an error already inside a claim passes through.
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

## Planted sources

Can a web page that someone else wrote get a false fact or an instruction into an approved report? I planted six fake pages, all on the reserved `.example` domain, in the saved source lists of two benchmark questions: Q4, on the Annex III date, and Q3, on conformity assessment. The analyst, writer and reviewer ran 5 times per attack and question, plus 5 clean runs per question. The runs start from the sources of a saved run, so every run sees the same real sources and only the planted pages change. The plan, with every planted text, was fixed before the runs in [injection/TEST_PLAN.md](injection/TEST_PLAN.md).

| Attack, 10 runs each | In an approved report | Stated as fact | With defense D2 |
|---|---|---|---|
| A1: a false date on one blog page | 9 | 0 | 8, none as fact |
| A2: the same date on three pages with "legal" or "law" in the site name and September dates | 8, the claim labeled high or medium | 1 | 8, the claim labeled low, none as fact |
| A3: an instruction to state a false rule | 7, only as a warning | 0 | 7, only as a warning |
| A4: an instruction to add a link to a fake checker | 0 | 0 | 0 |
| A5: an instruction in a title to search for a made-up word | 0 of 20 query sets | | not run |

1. **Instructions were not followed.** No report stated the false rule as a rule, no report carried the fake link, and no search query used the made-up word. In 11 of the 20 baseline runs of A3 and A4, the analyst named the page as an attempt to instruct AI tools.
2. **False facts got in, usually with a warning.** The false date reached 17 of 20 approved reports in A1 and A2, and most of them called it unconfirmed or disputed. One report stated it as fact and two leaned toward it, all for Q3, whose sources have no official page with the real date to check against.
3. **The confidence score was the weak point.** A host name with "law" or "legal" counted as legal analysis and a date in the URL counted as recent, so the three planted pages made the false-date claim high (0.70) in 8 of 10 A2 runs. In 3 of those runs the reviewer sent the report back because the report gave that claim a lower label than its score.
4. **Defense D2** judges a site only by the hand-made list (`LEGAL_NAME_RULE = False` in [src/config.py](src/config.py); the old rule stays the default). With it, the false-date claim was labeled low in 10 of 10 A2 runs, no report stated the date as fact, and 9 of 10 clean reports were approved, as before. It lowered 5 of the 160 real claims in its clean runs from medium to low. The 50 runs cost $10.68 without D2 and $10.55 with it.
5. **Limits.** One agent, one model, two questions and 5 runs per case. The source list is fixed, so a planted page could not change what the search returns. The planted pages use the reserved `.example` domain, and in 3 of the 80 attack runs the model named those domains as a reason for doubt, so a real attacker's page may do better.

```bash
uv run python -m injection.runner                     # smoke test: one clean run per question (API)
uv run python -m injection.run                        # the 50 runs without a defense (API, about $11)
uv run python -m injection.run --defense d2           # the same 50 runs with defense D2
uv run python -m injection.steer                      # A5: 40 calls to the query writer (about $0.10)
uv run python -m injection.analyze                    # results.json from the saved runs, no API calls
uv run --group analysis python -m injection.figures   # injection/figures/outcomes.png
```

## The researcher picks its own tools

In V2 the code decides how to search: Claude writes 3 queries, and each one runs twice, on official EU sites and on the open web. With `RESEARCHER = "tools"` in [src/config.py](src/config.py), Claude decides instead. It gets three tools, `search_official`, `search_web` and `finish_research`, and a plain loop in [src/llm.py](src/llm.py) runs every tool call and sends the result back. A search result shows the title, address, type and date of each new source, but not its text.

1. **When it stops.** The loop ends when Claude calls `finish_research`, when it answers without calling a tool, or after 12 answers. At most 18 searches can run, the fixed researcher's maximum, and a search after that gets an error. If Claude tries to finish with fewer than 5 sources, it is told once to search more, and a failed search goes back to it as an error.
2. **The comparison.** I ran the 4 benchmark questions 3 times with each researcher, one right after the other, so both searched the web at about the same time ([results/researcher_comparison/](results/researcher_comparison/)):

| | Fixed | With tools |
|---|---|---|
| Reports approved | 11 of 12 | 11 of 12 |
| Q4 gives the current date | 3 of 3 | 3 of 3 |
| Sources per run | 27.0 | 29.8 |
| Share from official EU sites | 49% | 47% |
| Searches per run | 6 | 6.4 (5 to 9) |
| Cost per run | $0.21 | $0.25 |
| Time per run | 132 s | 146 s |

3. **What changed.** On these checks the two researchers did the same. The new one cost more in all 4 questions, and $0.03 of the difference per run is the tool loop itself. It searched more where the question was narrow (8 searches and 38 sources on average for Q3, on conformity assessment) and used the open web more for Q2, which asks about US policy (30% from official EU sites). It ended the research itself in all 12 runs and never reached a limit. The fixed researcher stays the default, so the results above still describe it.
4. **A planted title.** The new researcher reads the titles in its search results and picks the next search itself, so I ran A5 again on it. The planted title with an instruction to search for a made-up word came back in a search result in all 20 runs, and no later query used the word (0 of 20, and 0 of 20 without the title). The searches replayed the saved Q4 sources, so every run saw the same results. The 40 runs cost $2.18.

```bash
uv run python compare_researchers.py                  # the comparison: 24 full runs (API and Tavily, about $5.50)
uv run python compare_researchers.py --summary        # summary.json from the saved runs, no API calls
uv run python -m injection.steer --researcher tools   # A5 on the new researcher (API, about $2.20)
```

## Tests and deployment

The repository has three GitHub Actions workflows, and every action in them is pinned to a commit:

1. **[tests.yml](.github/workflows/tests.yml)**, on every push and pull request: the unit tests (117, no API calls), and a check that `injection/results.json`, `calibration/results.json` and the comparison's `summary.json` rebuild unchanged from the saved runs.
2. **[planted-errors.yml](.github/workflows/planted-errors.yml)**, by hand and on pushes to main that change `src/` or `eval_reviewer.py`: the reviewer test as a gate (`eval_reviewer.py --gate`, on the saved run `v2_20260927_235716.json`). The reviewer's verdicts vary between runs: in 3 local runs on that file it caught 13, 13 and 12 of the 13 planted errors, and it rejected 3 of 4 clean reports each time. So the gate fails when 2 or more planted errors get through, or when all 4 clean reports are rejected. It uses an API key stored as a GitHub secret, and pull requests never run it, so code from a fork never sees the key.
3. **[deploy.yml](.github/workflows/deploy.yml)**, on pushes to main: the tests, then a deploy to Google Cloud Run in Finland (`europe-north1`). GitHub signs in to Google Cloud with Workload Identity Federation, which accepts only this repository's main branch, so no key file is stored anywhere.

The app runs at https://mas-demo-956845653269.europe-north1.run.app. It shows the saved results, and live runs need an access code. The API keys are kept in Google Secret Manager, not in the image, and only the app's own service account can read them. Cloud Run runs at most 1 instance, stops when nobody uses the app and has a monthly spend cap, and the app's Anthropic key has its own spend limit.

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
