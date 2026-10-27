"""The Researcher with tools: Claude picks its own searches.

Claude gets three tools, search_official, search_web and finish_research, and
decides which to call, with what query, and when to stop. Both search tools run
search_sources (src/nodes/researcher.py), the same search as the fixed
Researcher, so the Tavily settings, the source types and the snippets stay the
same. A search result tells Claude each new source's number, title, URL, type and
date. The snippets are not shown to Claude here; the Analyst reads them, as before.

Limits, set in code (src/config.py):
1. At most MAX_TOOL_SEARCHES searches. A search after that gets an error that
   asks Claude to finish.
2. At most MAX_TOOL_TURNS replies from Claude (the loop in src/llm.py).
3. A finish_research call with fewer than MIN_SOURCES sources is answered once
   with "find more sources" instead of ending the research.
A failed search goes back to Claude as a tool error.
"""

import logging
from datetime import date

from src.config import EFFORT, MAX_TOOL_SEARCHES, MAX_TOOL_TURNS, MIN_SOURCES, OFFICIAL_DOMAINS
from src.llm import LLMError, ToolResult, run_tool_loop
from src.nodes import researcher
from src.state import ResearchState, SourceDoc

logger = logging.getLogger(__name__)

SYSTEM = (
    "You research a question for a report on regulation. Today's date is {today}. "
    "Find sources with the search tools. Prefer primary sources, such as legal texts and "
    "official guidance, and recent developments. You can run at most {max_searches} searches. "
    "Find at least {min_sources} sources, then call finish_research when the sources found "
    "can answer the question, or when more searches add little."
)


def _search_tool(name: str, description: str) -> dict:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    }


TOOLS = [
    _search_tool(
        "search_official",
        f"Search only official EU sites ({', '.join(OFFICIAL_DOMAINS)}). Use it for legal texts, "
        "official guidance and official announcements. Returns the new sources found, each with "
        "its number, title, URL, source type and date.",
    ),
    _search_tool(
        "search_web",
        "Search the whole web: national authorities, law firms, news and blogs. Use it for "
        "interpretation, national rules, enforcement news and what official EU sites do not "
        "cover. Returns the new sources found, each with its number, title, URL, source type and date.",
    ),
    {
        "name": "finish_research",
        "description": "End the research. Call it when the sources found can answer the question, "
        "or when more searches add little.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "Why the research can end"}},
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
]


class Research:
    """One research run: the sources found and a log of every tool call."""

    def __init__(self, search=None):
        # Stands in for search_sources in the A5 test, which replays saved sources
        self.search = search or researcher.search_sources
        self.sources: list[SourceDoc] = []
        self.seen_urls: set[str] = set()
        self.queries: list[str] = []  # the queries that were searched
        self.calls: list[dict] = []
        self.told_to_continue = False
        self.finished = False
        self.turns: int | None = None
        self.ended_by = ""

    def run_tool(self, name: str, tool_input: dict, turn: int) -> ToolResult:
        """Run one tool call from Claude and log it (the run_tool of src.llm.run_tool_loop)."""
        call = {"turn": turn, "tool": name, **tool_input}
        if name == "finish_research":
            result = self._finish()
        elif name in ("search_official", "search_web"):
            result, new = self._search(name, tool_input["query"])
            call["urls"] = [s.url for s in new]
        else:
            result = ToolResult(f"Unknown tool: {name}", is_error=True)
        call["result"] = result.content.split("\n")[0]
        call["is_error"] = result.is_error
        self.calls.append(call)
        return result

    def _search(self, tool: str, query: str) -> tuple[ToolResult, list[SourceDoc]]:
        if len(self.queries) >= MAX_TOOL_SEARCHES:
            message = f"Search limit reached: all {MAX_TOOL_SEARCHES} searches are used. Call finish_research."
            return ToolResult(message, is_error=True), []
        self.queries.append(query)
        domains = OFFICIAL_DOMAINS if tool == "search_official" else None
        try:
            new = self.search(query, self.seen_urls, include_domains=domains, raise_errors=True)
        except researcher.SearchError as e:
            return ToolResult(f"The search failed: {e}", is_error=True), []

        first = len(self.sources) + 1
        self.sources += new
        found = f"{len(new)} new sources" if new else "No new sources: every result was found before"
        lines = [
            f"{found}. {len(self.sources)} sources so far, "
            f"{len(self.queries)} of {MAX_TOOL_SEARCHES} searches used."
        ]
        lines += [
            f"[{n}] {s.title} | {s.url} | {s.source_type} | {s.published_date or 'no date'}"
            for n, s in enumerate(new, first)
        ]
        return ToolResult("\n".join(lines)), new

    def _finish(self) -> ToolResult:
        if len(self.sources) < MIN_SOURCES and not self.told_to_continue:
            self.told_to_continue = True
            return ToolResult(
                f"Only {len(self.sources)} sources so far. Find at least {MIN_SOURCES} before you finish."
            )
        self.finished = True
        return ToolResult("Research finished.", stop=True)

    def summary(self) -> dict:
        return {
            "searches": len(self.queries),
            "turns": self.turns,
            "ended_by": self.ended_by,
            "told_to_continue": self.told_to_continue,
            "calls": self.calls,
        }


def research_with_tools(question: str, search=None) -> Research:
    """Let Claude research the question with the tools; `search` replaces the web search in tests."""
    research = Research(search)
    system = SYSTEM.format(
        today=date.today().isoformat(), max_searches=MAX_TOOL_SEARCHES, min_sources=MIN_SOURCES
    )
    try:
        loop = run_tool_loop(
            system,
            f"Research this question:\n\n{question}",
            TOOLS,
            research.run_tool,
            effort=EFFORT["researcher"],
            max_turns=MAX_TOOL_TURNS,
        )
    except LLMError as e:
        logger.warning(f"Research with tools stopped ({e}); keeping the {len(research.sources)} sources found")
        research.ended_by = f"error: {e}"
        return research
    research.turns = loop.turns
    research.ended_by = "finish_research" if research.finished else loop.ended_by
    return research


def tool_researcher_node(state: ResearchState) -> dict:
    """Claude searches with the tools until it is done; the graph then goes to the Analyst."""
    research = research_with_tools(state["question"])
    logger.info(
        f"Researcher with tools: {len(research.sources)} sources from {len(research.queries)} searches "
        f"in {research.turns} turns, ended by {research.ended_by}"
    )
    return {
        "search_queries": research.queries,
        "gathered_sources": research.sources,
        "iteration_count": 1,
        "research_complete": True,
        "tool_research": research.summary(),
        "current_phase": "researching",
    }
