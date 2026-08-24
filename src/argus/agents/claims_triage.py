"""
Milestone 5: Claims Triage -- last of the four specialists, and the first
agent that can hand off to a DIFFERENT specialist mid-conversation via
Command(goto=..., graph=Command.PARENT). This is the real fix for the
"flat router can't hold two intents" problem flagged since Milestone 1: a
top-level conditional edge only ever runs once, before any specialist has
looked closely at anything. A specialist mid-conversation, noticing a
second intent, is a fundamentally different routing moment -- and needs a
mechanism that can reach OUT of its own subgraph into the parent's.
"""

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from argus.compliance import build_summarize_node
from argus.harness import build_tools_node, human_escalation, route_after_tools, tools_present
from argus.llm import get_llm, get_token_usage
from argus.mcp_client import get_mcp_tools
from argus.state import State

CLAIMS_AGENT_SYSTEM_PROMPT = """You are the Claims Triage agent for \
Argus, a BFSI platform. When a claim is described, use get_claim_severity \
to assess it once you know the claim type -- never estimate severity \
yourself. If the claimant's account contains red flags (inconsistent \
details, suspicious timing, an implausible story), call \
flag_for_fraud_review instead of continuing normal triage -- do not try \
to assess fraud yourself, that's a different specialist's job."""

# flag_for_fraud_review IS offered to the model (so it can choose to call
# it), but is deliberately NOT included in the tools node's tool list --
# a call to it is intercepted inside call_model below and turned into a
# Command handoff, never actually "executed" as a normal tool. Both are
# still registered on the MCP server (mcp_server.py) so the model can
# discover/be offered flag_for_fraud_review at all -- it's just never
# passed to build_tools_node, so a real call to IT specifically never
# reaches the harness's execute_tools.
_tools = get_mcp_tools(names=["get_claim_severity", "flag_for_fraud_review"])
_llm_with_tools = get_llm().bind_tools(_tools)
_execute_tools = build_tools_node([t for t in _tools if t.name == "get_claim_severity"])
_summarize_decision = build_summarize_node("claims", ["get_claim_severity"])


def call_model(state: State) -> Command | dict:
    messages = [SystemMessage(CLAIMS_AGENT_SYSTEM_PROMPT)] + state["messages"]
    response = _llm_with_tools.invoke(messages)
    tokens = get_token_usage(response)

    handoff_call = next(
        (c for c in response.tool_calls if c["name"] == "flag_for_fraud_review"), None
    )
    if handoff_call:
        reason = handoff_call["args"].get("reason", "unspecified")
        print(f"[claims_triage] handing off to Fraud Investigation: {reason!r}")

        # Every tool_call in message history must eventually get a matching
        # ToolMessage, or the NEXT LLM call (inside Fraud's own agent node,
        # after the jump) can reject the history as malformed -- some
        # providers enforce this strictly. Ack it before jumping away,
        # don't just drop it because we're not "really" executing the tool.
        ack = ToolMessage(
            content="Flagged for fraud review -- handing this claim to Fraud Investigation.",
            tool_call_id=handoff_call["id"],
            name=handoff_call["name"],  # "flag_for_fraud_review" -- NOT a real data tool,
            # deliberately excluded from summarize_decision's real_tool_names check (compliance.py)
        )
        return Command(
            update={"messages": [response, ack], "total_tokens_used": tokens},
            goto="fraud",
            graph=Command.PARENT,
        )

    return {"messages": [response], "total_tokens_used": tokens}


def build_claims_agent():
    g = StateGraph(State)
    g.add_node("agent", call_model)
    g.add_node("tools", _execute_tools)
    g.add_node("human_escalation", human_escalation)
    g.add_node("summarize", _summarize_decision)

    g.add_edge(START, "agent")
    # "tools" or END normally; Command bypasses this router entirely on
    # handoff, so the remap below only ever applies to the non-handoff path.
    g.add_conditional_edges("agent", tools_present, {"tools": "tools", END: "summarize"})
    g.add_conditional_edges("tools", route_after_tools)
    g.add_edge("human_escalation", END)  # skips summarize -- no decision was actually reached
    g.add_edge("summarize", END)

    return g.compile()


if __name__ == "__main__":
    import asyncio

    async def main():
        app = build_claims_agent()

        print("=== Case 1: normal claim, no fraud signal ===")
        result = await app.ainvoke(
            {
                "messages": [("user", "I'd like to file a water damage claim from a burst pipe.")],
                "intent": "",
                "needs_escalation": False,
            }
        )
        for m in result["messages"]:
            print(f"[{m.type}]", m.content if m.content else m.tool_calls)

    asyncio.run(main())
