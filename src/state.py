from typing import Literal, Optional, TypedDict

from pydantic import BaseModel


class SourceDoc(BaseModel):
    """A single source retrieved by the Researcher."""

    url: str
    title: str
    snippet: str
    source_type: str  # official_eu, eu_guidance, national_authority, legal_analysis, industry, news_blog
    published_date: Optional[str] = None  # ISO date, when the search result or URL shows one
    search_query: str


class Confidence(BaseModel):
    """How a claim's confidence was computed (see src/confidence.py)."""

    score: float  # authority × recency × corroboration, capped at 1.0
    authority: float
    recency: float
    corroboration: float
    undated: bool  # the strongest source has no publication date


class Claim(BaseModel):
    """A factual claim extracted by the Analyst."""

    claim_id: str
    text: str
    supporting_sources: list[str]  # source URLs, all from the gathered sources
    category: str  # legal_requirement, timeline, enforcement, interpretation
    confidence: Confidence


class Contradiction(BaseModel):
    """A contradiction detected between sources."""

    claim_a: str
    claim_b: str
    source_a: str
    source_b: str
    description: str


class CheckResult(BaseModel):
    passed: bool
    notes: str


ReviewStatus = Literal["pending", "approved", "revision_requested", "not_approved", "review_failed"]


class ResearchState(TypedDict):
    # Input
    question: str

    # Researcher writes
    search_queries: list[str]
    gathered_sources: list[SourceDoc]
    iteration_count: int
    research_complete: bool

    # Analyst writes
    extracted_claims: list[Claim]
    contradictions: list[Contradiction]
    open_issues: list[str]

    # Writer writes
    report: str

    # Reviewer writes
    review_feedback: Optional[str]
    review_checklist: dict[str, CheckResult]
    review_status: ReviewStatus
    approved: bool
    review_count: int

    # Orchestration / UI
    current_phase: str
    error: Optional[str]


def initial_state(question: str) -> ResearchState:
    return {
        "question": question,
        "search_queries": [],
        "gathered_sources": [],
        "iteration_count": 0,
        "research_complete": False,
        "extracted_claims": [],
        "contradictions": [],
        "open_issues": [],
        "report": "",
        "review_feedback": None,
        "review_checklist": {},
        "review_status": "pending",
        "approved": False,
        "review_count": 0,
        "current_phase": "",
        "error": None,
    }
