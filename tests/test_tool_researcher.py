"""Tests for the Researcher with tools and the tool loop (no API calls).

A fake client stands in for Claude: it returns scripted replies with tool calls
and records every request. A fake Tavily client stands in for the web search.
"""

from itertools import count
from types import SimpleNamespace

import pytest

import src.llm as llm
import src.nodes.researcher as researcher
from injection import steer
from injection.planted import STEERING_PAGE, _official
from injection.runner import load_saved
from src.config import MAX_TOOL_SEARCHES, MAX_TOOL_TURNS, MIN_SOURCES, OFFICIAL_DOMAINS, RESULTS_PER_QUERY
from src.graph import build_graph
from src.llm import usage
from src.nodes.tool_researcher import research_with_tools, tool_researcher_node

_ids = count(1)


def reply(*calls, stop_reason=None, input_tokens=100, output_tokens=10):
    """A reply from Claude that calls tools, for example reply(("search_web", {"query": "q"}))."""
    content = [SimpleNamespace(type="thinking", thinking="", signature="sig")]
    content += [
        SimpleNamespace(type="tool_use", id=f"toolu_{next(_ids)}", name=name, input=tool_input)
        for name, tool_input in calls
    ]
    if not calls:
        content.append(SimpleNamespace(type="text", text="Done."))
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason or ("tool_use" if calls else "end_turn"),
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def search(query, tool="search_web"):
    return (tool, {"query": query})


FINISH = ("finish_research", {"reason": "enough"})


class FakeClient:
    """Returns the scripted replies in order and records every request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.messages = self  # so client.messages.create(...) works

    def create(self, **request):
        # A copy of the conversation as sent: the loop keeps adding to the same list
        self.requests.append(request | {"messages": list(request["messages"])})
        return self.replies.pop(0)

    def tool_results(self, turn: int) -> list[dict]:
        """The tool results sent back after Claude's reply in `turn` (1 is the first reply)."""
        return self.requests[turn]["messages"][-1]["content"]


class FakeTavily:
    """Returns `per_search` new results per search and records every search."""

    def __init__(self, per_search=3, fail_for=()):
        self.per_search = per_search
        self.fail_for = fail_for
        self.searches = []

    def search(self, **params):
        self.searches.append(params)
        if params["query"] in self.fail_for:
            raise TimeoutError("Tavily did not answer")
        n = len(self.searches)
        host = "ec.europa.eu" if params["include_domains"] else "ai-blog.com"
        return {
            "results": [
                {"url": f"https://{host}/{n}/{k}", "title": f"Page {n}.{k}", "content": "snippet text"}
                for k in range(self.per_search)
            ]
        }


@pytest.fixture
def fakes(monkeypatch):
    """Install a fake client with the given replies and a fake Tavily; returns both."""
    usage.reset()

    def install(replies, **tavily_options):
        client, tavily = FakeClient(replies), FakeTavily(**tavily_options)
        monkeypatch.setattr(llm, "_get_client", lambda: client)
        monkeypatch.setattr(researcher, "_get_tavily", lambda: tavily)
        return client, tavily

    return install


def test_each_tool_searches_the_right_sites(fakes):
    client, tavily = fakes([reply(search("AI Act text", "search_official"), search("AI Act news")), reply(FINISH)])
    research = research_with_tools("When do the rules apply?")

    assert [(s["query"], s["include_domains"]) for s in tavily.searches] == [
        ("AI Act text", OFFICIAL_DOMAINS),
        ("AI Act news", []),
    ]
    assert all(s["search_depth"] == "advanced" and s["max_results"] == RESULTS_PER_QUERY for s in tavily.searches)
    assert [s.source_type for s in research.sources] == ["eu_guidance"] * 3 + ["news_blog"] * 3
    assert research.queries == ["AI Act text", "AI Act news"]


def test_search_results_give_number_title_url_type_and_date_but_no_snippet(fakes):
    client, _ = fakes([reply(search("a")), reply(search("b")), reply(FINISH)])
    research_with_tools("q")

    second = client.tool_results(2)[0]["content"]
    assert second.startswith("3 new sources. 6 sources so far, 2 of 18 searches used.")
    assert "[4] Page 2.0 | https://ai-blog.com/2/0 | news_blog | no date" in second
    assert "snippet text" not in second


def test_claude_gets_its_whole_reply_back_with_the_results(fakes):
    first = reply(search("a"))
    client, _ = fakes([first, reply(FINISH)], per_search=5)
    research_with_tools("q")

    messages = client.requests[1]["messages"]
    assert messages[1] == {"role": "assistant", "content": first.content}  # thinking block included
    assert [r["tool_use_id"] for r in messages[2]["content"]] == [first.content[1].id]


def test_search_limit_holds(fakes):
    many = [search(f"q{i}") for i in range(MAX_TOOL_SEARCHES + 2)]
    client, tavily = fakes([reply(*many), reply(FINISH)])
    research = research_with_tools("q")

    assert len(tavily.searches) == MAX_TOOL_SEARCHES
    assert len(research.queries) == MAX_TOOL_SEARCHES
    refused = client.tool_results(1)[-2:]
    assert all(r["is_error"] and r["content"].startswith("Search limit reached") for r in refused)


def test_finish_research_ends_the_loop(fakes):
    client, _ = fakes([reply(search("a"), search("b")), reply(FINISH), reply(search("never asked"))])
    research = research_with_tools("q")

    assert len(client.requests) == 2
    assert research.ended_by == "finish_research" and research.turns == 2


def test_finishing_with_too_few_sources_is_refused_once(fakes):
    client, tavily = fakes(
        [reply(search("a")), reply(FINISH), reply(search("b")), reply(FINISH), reply(FINISH)], per_search=2
    )
    research = research_with_tools("q")

    told = client.tool_results(2)[0]
    assert told == {
        "type": "tool_result",
        "tool_use_id": told["tool_use_id"],
        "content": f"Only 2 sources so far. Find at least {MIN_SOURCES} before you finish.",
        "is_error": False,
    }
    # The second finish is accepted with 4 sources: Claude is told only once
    assert len(client.requests) == 4 and len(research.sources) == 4 < MIN_SOURCES
    assert research.told_to_continue and research.ended_by == "finish_research"


def test_failed_search_goes_back_as_a_tool_error(fakes):
    client, _ = fakes(
        [reply(search("timeout please"), search("fine")), reply(FINISH)], per_search=5, fail_for={"timeout please"}
    )
    research = research_with_tools("q")

    failed, worked = client.tool_results(1)
    assert failed["is_error"] and failed["content"] == "The search failed: Tavily did not answer"
    assert not worked["is_error"]
    assert research.queries == ["timeout please", "fine"]  # a failed search still counts


def test_usage_is_counted_for_every_turn(fakes):
    fakes(
        [reply(search("a"), input_tokens=1000, output_tokens=200), reply(FINISH, input_tokens=3000, output_tokens=50)],
        per_search=5,
    )
    research_with_tools("q")

    assert (usage.calls, usage.input_tokens, usage.output_tokens) == (2, 4000, 250)
    assert usage.cost_usd == pytest.approx((4000 * 2.0 + 250 * 10.0) / 1_000_000)


def test_turn_limit_ends_the_loop(fakes):
    client, _ = fakes([reply(search(f"q{i}")) for i in range(MAX_TOOL_TURNS + 1)])
    research = research_with_tools("q")

    assert len(client.requests) == MAX_TOOL_TURNS
    assert research.ended_by == "turn_limit" and research.turns == MAX_TOOL_TURNS


def test_reply_without_a_tool_call_ends_the_loop(fakes):
    client, _ = fakes([reply(search("a"), search("b")), reply()])
    research = research_with_tools("q")

    assert len(client.requests) == 2
    assert research.ended_by == "no_tool_call" and len(research.sources) == 6


def test_cut_off_reply_keeps_the_sources_found(fakes):
    fakes([reply(search("a")), reply(search("b"), stop_reason="max_tokens")])
    research = research_with_tools("q")

    assert research.ended_by.startswith("error: ") and len(research.sources) == 3


def test_every_tool_call_is_logged(fakes):
    fakes([reply(search("a", "search_official")), reply(FINISH)], per_search=5)
    research = research_with_tools("q")

    assert research.calls == [
        {
            "turn": 1,
            "tool": "search_official",
            "query": "a",
            "urls": [f"https://ec.europa.eu/1/{k}" for k in range(5)],
            "result": "5 new sources. 5 sources so far, 1 of 18 searches used.",
            "is_error": False,
        },
        {"turn": 2, "tool": "finish_research", "reason": "enough", "result": "Research finished.", "is_error": False},
    ]


def test_node_writes_the_sources_and_the_log_to_the_state(fakes):
    fakes([reply(search("a"), search("b")), reply(FINISH)])
    update = tool_researcher_node({"question": "q"})

    assert update["search_queries"] == ["a", "b"] and len(update["gathered_sources"]) == 6
    assert update["tool_research"]["searches"] == 2 and update["tool_research"]["ended_by"] == "finish_research"


def test_a5_replay_returns_saved_sources_with_the_planted_page_first_on_the_web():
    _, sources, _ = load_saved("q4")
    replay = steer.replay_search(sources, STEERING_PAGE)
    seen = set()

    official = replay("anything", seen, include_domains=OFFICIAL_DOMAINS)
    web = replay("anything", seen)
    assert [s.url for s in official] == [s.url for s in sources[:RESULTS_PER_QUERY]]  # the saved order
    assert web[0].url == STEERING_PAGE.url and web[0].search_query == "anything"
    assert [s.url for s in web[1:]] == [s.url for s in sources if not _official(s)][: RESULTS_PER_QUERY - 1]


def test_a5_counts_only_queries_written_after_the_planted_title_came_back(fakes):
    _, sources, _ = load_saved("q4")
    fakes([reply(search("Vantrelo, written blind"), search("dates")), reply(search("Vantrelo exemption")), reply(FINISH)])
    record = steer.tool_research_run("q", sources, planted=True)
    assert record["planted_shown_in_turn"] == 1 and record["canary_in_queries"]

    fakes([reply(search("Vantrelo, written blind"), search("dates")), reply(FINISH)])
    record = steer.tool_research_run("q", sources, planted=True)
    assert record["planted_shown_in_turn"] == 1 and not record["canary_in_queries"]


def _edges(graph) -> set[tuple]:
    return {(e.source, e.target) for e in graph.get_graph().edges}


def test_only_the_fixed_researcher_loops_in_the_graph():
    assert ("researcher", "researcher") in _edges(build_graph("fixed"))
    assert ("researcher", "researcher") not in _edges(build_graph("tools"))
    assert ("researcher", "analyst") in _edges(build_graph("tools"))
    with pytest.raises(KeyError):
        build_graph("other")
