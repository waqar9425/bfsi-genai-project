"""
Harness retry/escalation logic tested directly -- no LLM, no graph
traversal. Same testability principle as route_by_intent in Milestone 0:
the harness operates purely on a tool_calls list + tool functions, it has
no idea an LLM even exists, so we don't need one to test it.

Milestone 12: execute_tools is now async (MCP tools only support
.ainvoke()) -- these tests use plain LangChain @tool objects, not MCP
ones, but they still go through the SAME async execute_tools function, so
tests wrap calls in asyncio.run(). Plain @tool objects support .ainvoke()
out of the box (LangChain provides a default async wrapper over the sync
implementation when a tool has no native async one) -- verified by these
tests passing at all, not assumed.
"""

import asyncio

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from argus.harness import build_tools_node, route_after_tools


def _make_flaky_tool(fail_times: int):
    """A controllable fake tool: fails exactly `fail_times` times, then
    succeeds. Deliberately NOT random -- a flaky test is a bad test.
    """
    calls = {"count": 0}

    @tool
    def flaky_tool(x: int) -> dict:
        """A deliberately flaky tool, for testing harness retry logic."""
        calls["count"] += 1
        if calls["count"] <= fail_times:
            raise RuntimeError(f"simulated transient failure #{calls['count']}")
        return {"ok": True, "x": x}

    return flaky_tool, calls


def _ai_message_calling(tool_name: str):
    return AIMessage(
        content="", tool_calls=[{"name": tool_name, "args": {"x": 1}, "id": "call_1"}]
    )


def test_tool_succeeds_first_try_no_escalation():
    tool_fn, calls = _make_flaky_tool(fail_times=0)
    node = build_tools_node([tool_fn])

    state = {"messages": [_ai_message_calling("flaky_tool")], "intent": "", "needs_escalation": False}
    result = asyncio.run(node(state))

    assert result["needs_escalation"] is False
    assert calls["count"] == 1


def test_tool_recovers_after_one_retry_no_escalation():
    tool_fn, calls = _make_flaky_tool(fail_times=1)  # fails once, succeeds on the retry
    node = build_tools_node([tool_fn])

    state = {"messages": [_ai_message_calling("flaky_tool")], "intent": "", "needs_escalation": False}
    result = asyncio.run(node(state))

    assert result["needs_escalation"] is False  # the retry saved it
    assert calls["count"] == 2  # 1 failed attempt + 1 successful retry


def test_tool_exhausts_retry_budget_triggers_escalation():
    tool_fn, calls = _make_flaky_tool(fail_times=99)  # always fails
    node = build_tools_node([tool_fn])

    state = {"messages": [_ai_message_calling("flaky_tool")], "intent": "", "needs_escalation": False}
    result = asyncio.run(node(state))

    assert result["needs_escalation"] is True
    assert calls["count"] == 2  # exactly the retry budget, not more -- no infinite retry


def test_route_after_tools_escalates_when_flagged():
    state = {"messages": [], "intent": "", "needs_escalation": True}
    assert route_after_tools(state) == "human_escalation"


def test_route_after_tools_continues_when_ok():
    state = {"messages": [], "intent": "", "needs_escalation": False}
    assert route_after_tools(state) == "agent"


def test_execute_tools_reports_one_agent_turn():
    tool_fn, _ = _make_flaky_tool(fail_times=0)
    node = build_tools_node([tool_fn])
    state = {"messages": [_ai_message_calling("flaky_tool")], "intent": "", "needs_escalation": False}
    result = asyncio.run(node(state))
    assert result["agent_turns"] == 1


def test_route_after_tools_escalates_when_turn_budget_exhausted_even_if_tools_succeeded():
    # needs_escalation is False -- every tool call succeeded -- but the
    # request has already taken MAX_AGENT_TURNS (3) rounds. Should still
    # escalate: success on every individual call doesn't matter if the
    # loop never converges.
    state = {"messages": [], "intent": "", "needs_escalation": False, "agent_turns": 3}
    assert route_after_tools(state) == "human_escalation"


def test_route_after_tools_continues_when_under_turn_budget():
    state = {"messages": [], "intent": "", "needs_escalation": False, "agent_turns": 2}
    assert route_after_tools(state) == "agent"
