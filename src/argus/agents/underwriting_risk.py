"""
Milestone 4: Underwriting & Risk agent. Same ReAct shape as
fraud_investigation.py, but tool execution goes through the shared
harness (harness.py) instead of the bare ToolNode/tools_condition pair --
retry-then-escalate is now real, not just described in a docstring.

Milestone 12: get_risk_grade is now fetched from the Argus MCP server
(mcp_client.py) instead of imported directly -- see
fraud_investigation.py's docstring for the full "why does this now need
.ainvoke()" chain, identical here.
"""

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END

from argus.compliance import build_summarize_node
from argus.harness import build_tools_node, human_escalation, route_after_tools, tools_present
from argus.llm import get_llm, get_token_usage
from argus.mcp_client import get_mcp_tools
from argus.skills import REASON_CODE_SKILL
from argus.state import State

# Milestone 10: REASON_CODE_SKILL here is the EXACT SAME Skill object
# fraud_investigation.py imports -- not a similar instruction rewritten
# by hand in two places, genuinely shared, genuinely one source of truth.
UNDERWRITING_AGENT_SYSTEM_PROMPT = (
    "You are the Underwriting & Risk agent for Argus, a BFSI platform. "
    "When asked to assess credit or policy risk, use the get_risk_grade "
    "tool -- never estimate a grade yourself. If the user hasn't given "
    "you the loan amount or annual income, ask for whatever is missing "
    "before calling the tool.\n\n" + REASON_CODE_SKILL.instructions
)

_tools = get_mcp_tools(names=["get_risk_grade"])
_llm_with_tools = get_llm().bind_tools(_tools)
_execute_tools = build_tools_node(_tools)
_summarize_decision = build_summarize_node("underwriting", ["get_risk_grade"])


def call_model(state: State) -> dict:
    messages = [SystemMessage(UNDERWRITING_AGENT_SYSTEM_PROMPT)] + state["messages"]
    response = _llm_with_tools.invoke(messages)
    return {"messages": [response], "total_tokens_used": get_token_usage(response)}


def build_underwriting_agent():
    g = StateGraph(State)
    g.add_node("agent", call_model)
    g.add_node("tools", _execute_tools)
    g.add_node("human_escalation", human_escalation)
    g.add_node("summarize", _summarize_decision)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_present, {"tools": "tools", END: "summarize"})
    g.add_conditional_edges("tools", route_after_tools)  # "agent" or "human_escalation"
    # Deliberately NOT looping back to "agent" here: a failure that
    # survived one retry is, by definition, not something re-asking the
    # same model to re-call the same tool with the same bad args is going
    # to fix -- that's an infinite-loop risk for any deterministic failure
    # (e.g. bad input data), not just a missed opportunity. Escalation is
    # a handoff, not a retry -- the automated graph stops here.
    g.add_edge("human_escalation", END)  # skips summarize -- no decision was actually reached
    g.add_edge("summarize", END)

    return g.compile()


if __name__ == "__main__":
    import asyncio

    async def main():
        app = build_underwriting_agent()

        print("=== Case 1: enough info given up front ===")
        result = await app.ainvoke(
            {
                "messages": [
                    (
                        "user",
                        "What's the risk grade for a $180,000 loan against "
                        "$45,000 annual income?",
                    )
                ],
                "intent": "",
                "needs_escalation": False,
            }
        )
        for m in result["messages"]:
            print(f"[{m.type}]", m.content if m.content else m.tool_calls)

        print("\n=== Case 2: bad income data forces a retry, then escalation ===")
        # No checkpointer on THIS standalone graph -- interrupt() still
        # pauses cleanly (verified), but there's nothing to resume it
        # WITH here (no thread_id to come back to). graph.py's demo does
        # the full pause-AND-resume; this just confirms the pause fires.
        result2 = await app.ainvoke(
            {
                "messages": [
                    (
                        "user",
                        "Grade the risk for a $50,000 loan. Annual income is $0 "
                        "(unemployed, no income on file).",
                    )
                ],
                "intent": "",
                "needs_escalation": False,
            }
        )
        if "__interrupt__" in result2:
            print(f"  paused as expected: {result2['__interrupt__'][0].value}")
        else:
            for m in result2["messages"]:
                print(f"[{m.type}]", m.content if m.content else m.tool_calls)

    asyncio.run(main())
