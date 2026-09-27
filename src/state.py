from typing import TypedDict, Optional

from pydantic import BaseModel


class SourceDoc(BaseModel):
    """A single source retrieved by the Researcher."""

    url: str
    title: str
    snippet: str
    source_type: str  # official_eu, eu_guidance, national_authority, legal_analysis, industry, news_blog
    published_date: Optional[str] = None
    search_query: str


class Claim(BaseModel):
    """A factual claim extracted by the Analyst."""

    claim_id: str
    text: str
    supporting_sources: list[str]  # list of source URLs
    confidence: float  # 0.0 to 1.0
    category: str  # legal_requirement, timeline, enforcement, interpretation


class Contradiction(BaseModel):
    """A contradiction detected between sources."""

    claim_a: str
    claim_b: str
    source_a: str
    source_b: str
    description: str


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
    approved: bool
    review_count: int

    # Orchestration / UI
    current_phase: str
    error: Optional[str]
