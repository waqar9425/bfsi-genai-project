"""
Milestone 2 (ReAct loop) + Milestone 4 (harness-wrapped tool execution) +
Milestone 12 (real MCP tool transport).

Unlike the Milestone-1 orchestrator (forced, single structured-output call,
every time), this agent decides FOR ITSELF whether it has enough
information to call get_fraud_score, and can only give a final answer once
it has (or has explicitly asked the user for) what it needs.

Tool execution goes through the shared harness (harness.py) instead of the
bare ToolNode/tools_condition pair -- retry-then-escalate, same as
underwriting_risk.py. One harness, every specialist.

get_fraud_score is now fetched from the Argus MCP server (mcp_client.py)
instead of imported directly from tools/fraud_tools.py -- the tool's own
mock logic is completely unchanged, only how it's REACHED changed. This
is also why build_fraud_agent()'s compiled graph now requires .ainvoke()
throughout (see harness.py's module docstring for the full chain of why).
"""

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END

from argus.compliance import build_summarize_node
from argus.harness import build_tools_node, human_escalation, route_after_tools, tools_present
from argus.llm import get_llm, get_token_usage
from argus.mcp_client import get_mcp_tools
from argus.skills import FRAUD_NARRATIVE_SKILL, REASON_CODE_SKILL
from argus.state import State

# Milestone 10: the "explain plainly" and "SIU-note phrasing" instructions
# used to be hand-written inline here. Now they're shared, versioned
# fragments from skills.py -- REASON_CODE_SKILL is the SAME object
# underwriting_risk.py uses, not a similar-but-separately-maintained copy.
FRAUD_AGENT_SYSTEM_PROMPT = (
    "You are the Fraud Investigation agent for Argus, a BFSI platform. "
    "When asked to assess a claim for fraud, use the get_fraud_score "
    "tool -- never estimate a fraud score yourself. If the user hasn't "
    "given you the claim amount, prior claims count, or the hour it was "
    "filed, ask for whatever is missing before calling the tool.\n\n"
    + REASON_CODE_SKILL.instructions
    + "\n\n"
    + FRAUD_NARRATIVE_SKILL.instructions
)

_tools = get_mcp_tools(names=["get_fraud_score"])
_llm_with_tools = get_llm().bind_tools(_tools)
_execute_tools = build_tools_node(_tools)
_summarize_decision = build_summarize_node("fraud", ["get_fraud_score"])


def call_model(state: State) -> dict:
    """Every call re-prepends the system prompt rather than storing it in
    state -- state holds conversation *data*, the system prompt is agent
    *configuration*, so it doesn't belong in the persisted message history.
    """
    messages = [SystemMessage(FRAUD_AGENT_SYSTEM_PROMPT)] + state["messages"]
    response = _llm_with_tools.invoke(messages)
    return {"messages": [response], "total_tokens_used": get_token_usage(response)}


def build_fraud_agent():
    g = StateGraph(State)
    g.add_node("agent", call_model)
    g.add_node("tools", _execute_tools)
    g.add_node("human_escalation", human_escalation)
    g.add_node("summarize", _summarize_decision)

    g.add_edge(START, "agent")
    # Remap: tools_present still only ever returns "tools" or END, unchanged
    # -- but here END gets redirected to "summarize" instead of really
    # ending the subgraph. Verified this remapping works in Milestone 7.
    g.add_conditional_edges("agent", tools_present, {"tools": "tools", END: "summarize"})
    g.add_conditional_edges("tools", route_after_tools)  # "agent" or "human_escalation"
    g.add_edge("human_escalation", END)  # escalation skips summarize -- no decision was actually reached
    g.add_edge("summarize", END)

    return g.compile()


if __name__ == "__main__":
    import asyncio

    async def main():
        app = build_fraud_agent()

        print("=== Case 1: enough info given up front ===")
        result = await app.ainvoke(
            {
                "messages": [
                    (
                        "user",
                        "Can you check this claim for fraud? Amount is $14,500, "
                        "this is their 4th claim this year, filed at 2am.",
                    )
                ],
                "intent": "",
                "needs_escalation": False,
            }
        )
        for m in result["messages"]:
            print(f"[{m.type}]", m.content if m.content else m.tool_calls)

        print("\n=== Case 2: info missing, agent should ask first ===")
        result2 = await app.ainvoke(
            {
                "messages": [("user", "Can you check my last claim for fraud?")],
                "intent": "",
                "needs_escalation": False,
            }
        )
        for m in result2["messages"]:
            print(f"[{m.type}]", m.content if m.content else m.tool_calls)

    asyncio.run(main())
