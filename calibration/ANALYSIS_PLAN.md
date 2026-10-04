# Calibration of the claim confidence: analysis plan

Written on 4 October 2026, before any claim was labeled. The rules below are fixed. If a rule has to change after labeling starts, the old text stays and the change is added under "Changes after labeling started", with the reason.

## The question

The agent gives every claim a confidence score from 0 to 1, computed in code from the sources that support the claim: authority × recency × corroboration, capped at 1.0 (`src/confidence.py`). The report shows it as a label: high from 0.7, medium from 0.4 and low below 0.4. The score was built to measure the evidence, and it was not built as a probability. This analysis checks how well it predicts whether a claim is correct:

1. **Ranking:** are high-confidence claims correct more often than low-confidence ones?
2. **Calibration:** read as a probability, does a score of about 0.7 mean correct about 70% of the time?
3. **Decision:** if every claim below a cut-off went to a person, how many claims would the person check, and how many wrong claims would still pass unchecked?
4. **Other signals:** does the model's own stated confidence, or its agreement across repeated checks, do better than the score?

## Data and scope

1. `results/v2_20260927_235716.json` (V2, 4 questions, 75 claims) and `results/v3_20260927_235419.json` (V3, 4 questions, 81 claims). Together they have 156 claims, of which 66 are high, 35 medium and 55 low.
2. Left out: the `*_first_query_only.json` files and `results/archive-2026-03/`, which come from older versions of the agent, and V1, which has no claims or scores.
3. One row is one scored claim. Claim IDs start again for each question, so the key is the run, the question and the claim ID, for example `V2-Q1-claim_001`.
4. V2 and V3 answered the same four questions, so some claims say the same thing. Both are kept, because each has its own score. The claims come from 8 reports and the claims in one report share sources, so they are not independent. The intervals below treat them as independent, which makes them narrower than they should be.

## Label rules

1. Truth is judged as of the run date, 27 September 2026, against official sources only:
   1. EUR-Lex: Regulation (EU) 2024/1689 (the AI Act) and Regulation (EU) 2026/1744, which amended it and entered into force on 27 July 2026.
   2. The European Commission's AI Act pages and the AI Act Service Desk.
   3. The NANDO database, for notified bodies.
   4. Official US government pages, such as whitehouse.gov and federalregister.gov, for US policy claims.
   5. Official pages of other EU institutions and of national governments, only for facts about their own actions, for example a vote, a law they adopted or an authority they designated.
2. The agent's own sources are never used as evidence. An official page is read directly, and never through the agent's snippet of it.
3. Each claim gets one of three labels:
   1. `correct`: an official source confirms all of the claim.
   2. `incorrect`: an official source contradicts any part of the claim. A claim that is partly wrong is `incorrect`, with `partly_correct` set to `yes`, so that the count can be reported.
   3. `unverifiable`: no official source settles it. Examples are "no enforcement actions so far", and claims that are only an interpretation or a forecast.
4. `interpretation` claims are labeled `correct` or `incorrect` only when they state a checkable fact. Otherwise they are `unverifiable`.
5. A claim about the Act as first adopted, for example "as originally adopted" or "under the 2024 text", is checked against the 2024 text. A claim about the law today is checked against the text as amended by Regulation (EU) 2026/1744.
6. When a claim reports what someone said, for example "the Commission states that ...", the fact it reports is checked. If it only reports an opinion or an expectation, it is `unverifiable`.
7. A claim that leaves out a detail is not wrong for that reason, and a paraphrase is fine when the meaning is the same. A detail that is stated wrongly makes the claim `incorrect`, and this includes a wrong article number or date.
8. A claim that can be read in two ways is judged on the reading a normal reader would take, and the note says which reading was used.
9. A claim that something does not exist, for example "no notified bodies have been designated", is checked only where an official register would list it, which here means NANDO for notified bodies. Otherwise it is `unverifiable`.
10. Every label records the official URL and the exact sentence or article that decides it.

Rules 1.1 to 1.4, 2 to 4 and 10 are the first version. Rules 1.5 and 5 to 9 were added on 4 October 2026, before labeling, so that common cases are decided by a rule and not case by case.

## How the labels are made

1. The labels are drafted from a blind copy, `calibration/labels.csv`, which has only an ID, the question, the category and the claim text. It has no score, confidence label, authority, recency, corroboration, run, supporting URLs or source types. The URLs are hidden because they show the source type, and the source type sets most of the score. The rows are shuffled within each question, so that the order does not show the run.
2. The drafts are written by Claude Opus 5.5, each with its evidence. The agent runs on Claude Sonnet 5, which is from the same model family, so every draft is checked by hand against its evidence before it is used, and only checked labels go into the results. The file keeps the draft label next to the final label, so that the number of changed labels can be reported.
3. The review starts with the hardest cases: drafts marked as a judgment call first, then `incorrect`, then `unverifiable`, then `correct`.
4. Before labeling, the drafter had seen the overall spread of scores (not per claim), the text and URLs of the first claim for question 4 in each run, and the two claims discussed in the earlier blog post (the amendment's number and the "36 months" error).

## Metrics

Accuracy is computed on the claims labeled `correct` or `incorrect`. The `unverifiable` claims are counted for each confidence label and left out of the accuracy numbers.

1. **Per confidence label:** the number of claims, the number unverifiable, the number correct, the share correct with a 95% Wilson interval, and the mean score.
2. **Reliability diagram:** the mean score against the share correct in each score bin. The bins are the three labels, with edges at 0.4 and 0.7. Finer bins, with edges at 0.15, 0.4, 0.7 and 0.9, are shown only if every finer bin keeps at least 15 labeled claims. These edges were chosen from the spread of the scores, before any claim was labeled.
3. **Calibration error:** the expected calibration error (ECE) is the gap between the mean score and the share correct in each bin, averaged over the bins and weighted by the number of claims in each bin. The Brier score is the mean squared difference between the score and the outcome (1 for correct, 0 for incorrect), and it is shown next to the Brier score of a constant guess equal to the overall share correct. With about 150 claims both numbers are rough, and they are reported as rough.
4. **Ranking:** the AUROC of the score for correct against incorrect claims, with tied scores counted as half, and a 95% bootstrap interval from 2,000 resamples of the claims with a fixed seed.
5. **Decision table** for the cut-offs 0.4 and 0.7, where every claim below the cut-off goes to a person: the share of all 156 claims sent to a person, and the number and share of incorrect claims that pass without a check. "Check everything" and "check nothing" are shown as reference rows.
6. **Breakdowns** by the type of the strongest source (the source that set the authority), by category and by run, each only for groups with at least 15 labeled claims.

## Other uncertainty signals

For every claim, `claude-sonnet-5` gets the claim and its cited sources in the format the Analyst saw them (`_format_sources` in `src/nodes/analyst.py`: title, URL, source type, date and snippet), through `src/llm.py`, with effort "medium" as for the Analyst. The system prompt gives the run date, 27 September 2026.

1. **Stated confidence:** asked once for the probability, from 0 to 100, that the claim is correct.
2. **Agreement:** asked 5 times, in separate calls, whether the sources fully support the claim, yes or no. The signal is the share of yes answers.

The model takes no temperature setting, so the repeated answers can only vary through its default sampling. The share of claims whose 5 answers are not all the same is reported, and if the answers almost never differ, that is reported as a finding.

Both signals get the metrics above, with the same bins and cut-offs. The signals do not have the same spread as the score, so they are also compared at equal workload: each signal sends the same number of claims to a person as the score does at 0.4 and at 0.7, and the incorrect claims that pass are counted. The difference in AUROC between the score and each signal gets a paired bootstrap interval. The cost is estimated before the run.

## Recalibration (optional)

If there is time: a logistic regression on the score (Platt scaling), with 5-fold cross-validation. The calibration error and the Brier score are reported on the held-out folds only, so the same claims are never used both to fit and to report.

## What is reported

All numbers go into `calibration/results.json`, which `calibration/analyze.py` computes from `calibration/claims.csv` without API calls. Every number in the README, the blog post and any other write-up comes from that file.

## Changes after labeling started

Two clarifications, written on 4 October 2026 during the first labeling pass. No rule above was changed.

1. Some claims describe what other, outdated sources say and mark them as outdated, for example "older sources state that the obligations apply from 2 August 2026". These claims are about the sources and not about the law, and no official source settles what other sources say, so they are `unverifiable`. Rule 6 covers claims that use a source to state a fact, and those are still checked on the fact.
2. For rule 4, a claim is labeled on its checkable facts, and descriptive words such as "primary", "most" or "strict" are set aside. When the main point of a claim is itself an interpretation, for example that two definitions are "closely aligned", the claim is `unverifiable` even if it also mentions a checkable detail.

One change to the method, also made on 4 October 2026 and before any score was looked at:

3. The plan said that every draft is checked by hand. Instead, the drafts were checked in a second pass by the same model that wrote them, rule by rule, and the final labels were fixed at that point. The second pass changed 1 of the 156 labels, because an Irish regulation had been amended in July 2026. The drafter and the reviewer are the same model, so this is not an independent check. A random sample of 20 labels (seed 20261004) is checked by hand, blind to the scores, and the share of agreement is reported. Any write-up says that the labels were drafted and checked by an AI model, with a check by hand on a sample.

One analysis added after the first results, on 4 October 2026:

4. A sensitivity check repeats the main numbers without the 35 claims marked as judgment calls during labeling, to see whether the borderline labels drive the result. It is reported as an extra check and not as part of the plan (`sensitivity_without_judgment_calls` in `results.json`).
