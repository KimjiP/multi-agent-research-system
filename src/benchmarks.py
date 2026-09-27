"""Benchmark questions shared by V1, V2 and V3, and the automatic checks run on every answer."""

import re

BENCHMARK_QUESTIONS = [
    # Q1: Specific compliance scenario (tests practical reasoning)
    "What are the current compliance requirements for a company deploying a high-risk AI hiring "
    "tool under the EU AI Act, and what enforcement actions have been taken so far?",
    # Q2: Cross-jurisdictional comparison (tests breadth; US Executive Order 14110 was revoked
    # in January 2025 and must not be presented as current policy)
    "How do the EU AI Act's requirements for foundation model providers compare to current US "
    "federal AI policy approaches?",
    # Q3: Narrow technical question (tests depth and specificity)
    "What conformity assessment procedures are required for high-risk AI systems under the EU AI "
    "Act, and which notified bodies have been designated so far?",
    # Q4: A date that changed in 2026 (tests whether the answer is current)
    "When do the obligations for high-risk AI systems listed in Annex III of the EU AI Act start "
    "to apply?",
]

# As of September 2026 the Annex III obligations apply from 2 December 2027: the Digital
# Omnibus on AI (Regulation (EU) 2026/1744, in force 27 July 2026) moved them from 2 August 2026.
_CURRENT_ANNEX_III_DATE = re.compile(r"December\s+2,?\s+2027|2\s+December\s+2027|December\s+2027", re.I)


def mentions_current_annex_iii_date(text: str) -> bool:
    """True if the text gives the Annex III application date as amended in 2026."""
    return bool(_CURRENT_ANNEX_III_DATE.search(text))
