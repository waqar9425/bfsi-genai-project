"""
Milestone 6: Policy & Customer -- the last of the four specialists, and
the first one grounded in RETRIEVAL rather than a computed number. Same
generic agent/tools/harness shape as Fraud and Underwriting (see Layer 2
of the architecture walkthrough) -- what's different is entirely inside
the tool and the system prompt, not the graph shape.
"""

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END

from argus.compliance import build_summarize_node
from argus.harness import build_tools_node, human_escalation, route_after_tools, tools_present
from argus.llm import get_llm, get_token_usage
from argus.mcp_client import get_mcp_tools
from argus.skills import COVERAGE_LOOKUP_SKILL
from argus.state import State

POLICY_AGENT_SYSTEM_PROMPT = (
    "You are the Policy & Customer agent for Argus, a BFSI platform. "
    "Answer coverage, claims-process, and account questions using ONLY "
    "the search_policy_docs tool -- never answer from general insurance "
    "knowledge, this company's policy wording is specific and may differ "
    "from industry norms.\n\n" + COVERAGE_LOOKUP_SKILL.instructions
)

_tools = get_mcp_tools(names=["search_policy_docs"])
_llm_with_tools = get_llm().bind_tools(_tools)
_execute_tools = build_tools_node(_tools)
_summarize_decision = build_summarize_node("policy", ["search_policy_docs"])


def call_model(state: State) -> dict:
    messages = [SystemMessage(POLICY_AGENT_SYSTEM_PROMPT)] + state["messages"]
    response = _llm_with_tools.invoke(messages)
    return {"messages": [response], "total_tokens_used": get_token_usage(response)}


def build_policy_agent():
    g = StateGraph(State)
    g.add_node("agent", call_model)
    g.add_node("tools", _execute_tools)
    g.add_node("human_escalation", human_escalation)
    g.add_node("summarize", _summarize_decision)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_present, {"tools": "tools", END: "summarize"})
    g.add_conditional_edges("tools", route_after_tools)
    g.add_edge("human_escalation", END)  # skips summarize -- no decision was actually reached
    g.add_edge("summarize", END)

    return g.compile()


if __name__ == "__main__":
    import asyncio

    async def main():
        app = build_policy_agent()

        for user_input in [
            "Is water damage from a burst pipe covered?",
            "How do I cancel my policy and will I get money back?",
            "Do you cover damage from a meteor strike?",  # deliberately not in the corpus
        ]:
            print(f"\n--- input: {user_input!r} ---")
            result = await app.ainvoke(
                {"messages": [("user", user_input)], "intent": "", "needs_escalation": False}
            )
            for m in result["messages"]:
                print(f"[{m.type}]", m.content if m.content else m.tool_calls)

    asyncio.run(main())
