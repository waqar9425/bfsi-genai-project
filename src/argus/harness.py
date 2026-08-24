"""
The harness: a thin operational layer wrapping tool execution, shared by
every specialist agent -- independent of any one agent's own logic (see
blueprint Section 06). Three jobs:

1. Retry: one retry on a failed tool call, then give up -- never loop
   forever, never silently swallow a persistent failure.
2. Escalate, don't fabricate: if a tool still fails after its retry
   budget, the graph routes to a human-escalation node instead of letting
   the LLM guess at an answer. `needs_escalation` in State is the signal.
3. Turn budget (Milestone 9): cap how many tool-calling rounds a single
   request can take, even when every individual call SUCCEEDS -- protects
   against a model that just keeps calling tools in circles without ever
   converging on an answer. A softer, business-aware sibling of
   LangGraph's own recursion_limit (blunter, graph-wide, hard-crashes
   instead of escalating gracefully -- see patterns/recursion_limit.py).

Built as plain functions any agent's graph can wire in, replacing the bare
`ToolNode` / `tools_condition` pair from Milestone 2 with harness-wrapped
equivalents -- same slot in the graph, more production-shaped behavior.

Milestone 12: execute_tools is now ASYNC. Tools are MCP-sourced
(mcp_client.py) as of this milestone, and MCP-backed LangChain tool
objects only support .ainvoke() -- verified directly: calling .invoke()
on one raises `NotImplementedError: StructuredTool does not support sync
invocation.` That one fact propagates further than it looks: LangGraph
requires the WHOLE containing graph to be run via .ainvoke() the moment
ANY node in it is async -- verified this holds even through nested
subgraphs (every specialist here is one, nested in the parent
orchestrator) -- so every specialist's graph, and the top-level graph.py,
now requires .ainvoke() throughout. Sync nodes (call_model,
classify_intent, summarize_decision, ...) don't need to change themselves
-- verified mixed sync+async graphs run fine under .ainvoke() -- only
whoever ACTUALLY invokes the compiled graph needs to switch.
"""

import json

from langchain_core.messages import ToolMessage
from langgraph.graph import END
from langgraph.types import interrupt

from argus.skills import ESCALATION_SKILL
from argus.state import State

MAX_ATTEMPTS = 2  # 1 initial try + 1 retry, per tool call
MAX_AGENT_TURNS = 3  # total tool-calling rounds allowed per REQUEST (see state.py's agent_turns)


async def _invoke_with_retry(tool_fn, call: dict) -> tuple[ToolMessage, bool]:
    """Returns (tool_message, failed).

    Milestone 12 correction, caught LIVE (not reasoned out in advance): an
    MCP tool raising an exception does NOT propagate as a Python exception
    through .ainvoke() -- FastMCP catches it server-side and returns a
    normal-looking result; langchain-mcp-adapters converts that into a
    ToolException internally, but that too gets caught by an internal
    handle_tool_error callback and turned back into ordinary content. A
    naive try/except around .ainvoke() NEVER fires for a real MCP tool
    error -- verified directly: the first version of this function did
    exactly that, and a live escalation test that used to pause silently
    stopped pausing at all, because "failed" was never set to True.

    The only reliable signal is `ToolMessage.status`, and getting a real
    ToolMessage back (not just bare content) requires invoking with the
    FULL tool-call dict -- {"name", "args", "id", "type": "tool_call"},
    exactly the shape response.tool_calls already gives us -- not just the
    bare args dict. Verified: bare args -> raw content blocks, no status
    at all; full call dict -> a proper ToolMessage with
    status="success"/"error".

    BUT a plain (non-MCP) tool with no handle_tool_error configured
    behaves differently AGAIN -- verified directly: it raises a real
    Python exception through .ainvoke() rather than swallowing it. So
    both mechanisms are needed together: try/except for plain tools and
    genuine transport/connection failures, AND a .status check for MCP's
    internally-caught tool errors. Neither alone covers both cases.
    """
    last_failure_message = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            message = await tool_fn.ainvoke(call)
        except Exception as exc:  # noqa: BLE001 -- intentionally broad: any tool failure must be caught here, not crash the graph
            print(f"[harness] {tool_fn.name} failed (attempt {attempt}/{MAX_ATTEMPTS}): {exc!r}")
            last_failure_message = ToolMessage(
                content=json.dumps({"error": str(exc)}), tool_call_id=call["id"], name=call["name"]
            )
            continue

        if getattr(message, "status", "success") != "error":
            return message, False
        print(f"[harness] {tool_fn.name} failed (attempt {attempt}/{MAX_ATTEMPTS}): {message.content!r}")
        last_failure_message = message

    return last_failure_message, True


def build_tools_node(tools: list):
    """Factory: given a specialist's tool list, returns an ASYNC node
    function that executes every tool call in the last message through
    the retry wrapper above, and sets `needs_escalation` if any of them
    exhausted their retry budget.
    """
    tools_by_name = {t.name: t for t in tools}

    async def execute_tools(state: State) -> dict:
        last_message = state["messages"][-1]
        tool_messages = []
        escalate = False

        for call in last_message.tool_calls:
            tool_fn = tools_by_name[call["name"]]
            tool_message, failed = await _invoke_with_retry(tool_fn, call)
            if failed:
                escalate = True
            tool_messages.append(tool_message)

        # Milestone 11: was `"agent_turns": 1` relying on an operator.add
        # reducer -- checkpointing exposed why that's wrong (see state.py).
        # Explicit read-then-increment instead: last-write-wins, correctly
        # scoped to what THIS turn has done so far, not the whole thread.
        turns_so_far = state.get("agent_turns", 0) + 1
        return {"messages": tool_messages, "needs_escalation": escalate, "agent_turns": turns_so_far}

    return execute_tools


def route_after_tools(state: State) -> str:
    """Post-tools router: escalate to a human instead of looping back to
    the agent if EITHER a tool call exhausted its retry budget OR this
    request has taken too many tool-calling rounds already -- two
    different triggers, same escalation destination. The agent never even
    sees a chance to paper over a failure with a guess, and never gets to
    loop indefinitely even when every individual call is succeeding.
    """
    if state.get("needs_escalation"):
        return "human_escalation"
    if state.get("agent_turns", 0) >= MAX_AGENT_TURNS:
        print(
            f"[harness] turn budget exhausted ({state['agent_turns']}/{MAX_AGENT_TURNS}) "
            "-- escalating"
        )
        return "human_escalation"
    return "agent"


def human_escalation(state: State) -> dict:
    # Milestone 11: this used to just return a canned message and end the
    # turn immediately -- a dead end, described as such since Milestone 3.
    # interrupt() genuinely PAUSES the whole graph here -- verified
    # empirically that this works even though human_escalation lives
    # inside a nested specialist subgraph, not the top-level graph -- and
    # waits for Command(resume=<decision>) before continuing. Whatever
    # code called .invoke() gets back state["__interrupt__"] describing
    # what's pending, instead of a normal final answer.
    #
    # CAUTION, verified directly: on resume, this function re-runs from
    # the TOP, not from the interrupt() call -- interrupt() just returns a
    # value the second time instead of pausing again. Any code before it
    # in this function would run TWICE. Nothing here does, but it's a real
    # gotcha for any node this pattern gets copied into later.
    human_decision = interrupt(
        {
            "reason": "automated processing failed or exceeded its turn budget",
            "needs_escalation": state.get("needs_escalation", False),
            "agent_turns": state.get("agent_turns", 0),
        }
    )
    reply = f"{ESCALATION_SKILL.instructions} A human reviewed this and said: {human_decision}"
    return {"messages": [("ai", reply)], "needs_escalation": False}


def tools_present(state: State) -> str:
    """Same job as LangGraph's prebuilt `tools_condition`, reimplemented
    here so the whole tool-or-not routing decision lives next to the rest
    of the harness rather than split across two import sources.
    """
    last_message = state["messages"][-1]
    has_calls = bool(getattr(last_message, "tool_calls", None))
    return "tools" if has_calls else END
