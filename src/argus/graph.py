"""
Milestone 3+4+5+6+8+9+11 graph: all four specialists are real, every one
flows through "compliance" before ending (Send-based fan-out, Milestone
7+8), Claims can hand off to "fraud" directly via Command (Milestone 5),
every request passes through PII redaction first (Milestone 9), and now
(Milestone 11) the whole graph is checkpointed -- conversations persist
across separate .invoke() calls on the same thread_id, and
human_escalation is a real pause-and-resume point (see harness.py).
"""

from argus.agents.claims_triage import build_claims_agent
from argus.agents.fraud_investigation import build_fraud_agent
from argus.agents.policy_customer import build_policy_agent
from argus.agents.underwriting_risk import build_underwriting_agent
from argus.compliance import dispatch_to_compliance, log_decision
from argus.guardrails import redact_pii_node
from argus.llm import get_llm, get_token_usage
from argus.schemas import IntentClassification
from argus.state import State
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

ORCHESTRATOR_SYSTEM_PROMPT = """You are the routing layer for Argus, a \
BFSI (banking/insurance) agent platform. Read the user's message and \
classify it into exactly one intent so it can be routed to the right \
specialist agent. Be decisive even on ambiguous messages -- pick the \
single best-fitting intent and explain your reasoning briefly."""

_llm = get_llm()
# include_raw=True: without it, .invoke() returns ONLY the parsed
# IntentClassification object, with no way to read token usage at all
# (verified empirically, Milestone 9) -- also changes a parse failure
# from an immediate crash into an object you can inspect and handle.
_structured_llm = _llm.with_structured_output(IntentClassification, include_raw=True)


def classify_intent(state: State) -> dict:
    """Real Orchestrator node: an LLM call constrained to return
    IntentClassification via tool-calling under the hood (see Lesson 1).
    """
    last_user_msg = state["messages"][-1]
    response = _structured_llm.invoke(
        [("system", ORCHESTRATOR_SYSTEM_PROMPT), last_user_msg]
    )
    result: IntentClassification = response["parsed"]
    tokens = get_token_usage(response["raw"])
    print(f"[classify_intent] intent={result.intent!r} reasoning={result.reasoning!r} tokens={tokens}")
    return {"intent": result.intent, "total_tokens_used": tokens}


def route_by_intent(state: State) -> str:
    return state["intent"]


# Compiled once at module load, same reuse-not-rebuild reasoning as the
# LLM clients in llm.py (see Milestone 1, drill question 3).
_fraud_agent = build_fraud_agent()
_underwriting_agent = build_underwriting_agent()
_claims_agent = build_claims_agent()
_policy_agent = build_policy_agent()


def build_graph():
    g = StateGraph(State)

    g.add_node("redact_pii", redact_pii_node)
    g.add_node("classify_intent", classify_intent)
    g.add_node("fraud", _fraud_agent)  # a compiled graph, used directly as a node
    g.add_node("claims", _claims_agent)  # can hand off straight to "fraud" via Command
    g.add_node("underwriting", _underwriting_agent)  # also a compiled graph, used as a node
    g.add_node("policy", _policy_agent)  # the RAG-grounded specialist
    g.add_node("compliance", lambda s: {})  # pass-through -- all the work is in the conditional edge below
    g.add_node("log_decision", log_decision)

    g.add_edge(START, "redact_pii")
    g.add_edge("redact_pii", "classify_intent")

    g.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "fraud": "fraud",
            "claims": "claims",
            "underwriting": "underwriting",
            "policy": "policy",
        },
    )

    # Every specialist now flows through compliance before actually ending,
    # whether reached via classify_intent's normal routing OR via Claims'
    # Command handoff landing on "fraud" directly -- this edge applies to
    # "fraud" regardless of how execution arrived there.
    g.add_edge("fraud", "compliance")
    g.add_edge("claims", "compliance")
    g.add_edge("underwriting", "compliance")
    g.add_edge("policy", "compliance")

    g.add_conditional_edges("compliance", dispatch_to_compliance)  # Send-based fan-out, no mapping needed
    g.add_edge("log_decision", END)

    # MemorySaver: in-process, lost on restart -- fine for dev/learning.
    # A real deployment would use SqliteSaver/PostgresSaver so
    # conversations survive a process restart. Swapping which one touches
    # only this line, nothing about the graph itself changes.
    return g.compile(checkpointer=MemorySaver())


def _print_result(result: dict) -> None:
    for m in result["messages"]:
        print(" ", m.type, "->", m.content)
    print(
        f"  [budget] agent_turns={result.get('agent_turns', 0)} "
        f"total_tokens_used={result.get('total_tokens_used', 0)}"
    )


if __name__ == "__main__":
    import asyncio

    from langgraph.types import Command

    async def main():
        # Milestone 12: every specialist's tools node is now async (MCP-
        # backed), which propagates all the way up through the nested
        # subgraphs to this top-level graph -- .ainvoke() throughout,
        # verified this is required, not optional (Section on why in
        # harness.py's module docstring).
        app = build_graph()

        print("=" * 70)
        print("DEMO 1: multi-turn memory -- same thread_id across two separate")
        print("app.ainvoke() calls, only the NEW message passed each time.")
        print("=" * 70)
        memory_config = {"configurable": {"thread_id": "demo-memory-thread"}}

        print("\n--- turn 1 ---")
        r1 = await app.ainvoke(
            {"messages": [("user", "My name is Dana. What's the risk grade for a $50,000 loan against $100,000 income?")],
             "intent": "", "needs_escalation": False},
            config=memory_config,
        )
        _print_result(r1)

        print("\n--- turn 2 (same thread -- does it remember my name?) ---")
        r2 = await app.ainvoke({"messages": [("user", "What's my name?")]}, config=memory_config)
        _print_result(r2)

        print("\n" + "=" * 70)
        print("DEMO 2: real human-in-the-loop -- trigger a genuine escalation")
        print("(bad income data), catch the interrupt, then resume as a human would.")
        print("=" * 70)
        hitl_config = {"configurable": {"thread_id": "demo-hitl-thread"}}

        print("\n--- first invoke: should PAUSE at human_escalation, not finish ---")
        r3 = await app.ainvoke(
            {
                "messages": [
                    ("user", "Grade the risk for a $50,000 loan. Annual income is $0 (unemployed).")
                ],
                "intent": "",
                "needs_escalation": False,
            },
            config=hitl_config,
        )
        pending = r3.get("__interrupt__")
        print(f"  paused: {'yes' if pending else 'NO -- expected a pause here'}")
        if pending:
            print(f"  interrupt payload: {pending[0].value}")

        print("\n--- resuming, as if a human underwriter reviewed the case ---")
        r4 = await app.ainvoke(
            Command(resume="Manually verified income via pay stubs -- proceed as grade C."),
            config=hitl_config,
        )
        _print_result(r4)

    asyncio.run(main())
