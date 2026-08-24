"""
Milestone 8: Compliance & Audit -- structured per-agent decisions, fanned
out to a shared audit log via Send (Milestone 7's fan-out pattern, now
applied for real instead of a toy demo).

Two pieces, used in two different places:
  build_summarize_node(agent_name) -- a factory, one instance per
    specialist, inserted right before that specialist's subgraph reaches
    its own END. Produces a typed AgentDecision from the conversation.
  dispatch_to_compliance / log_decision -- live in the PARENT graph
    (graph.py), fanning out over whatever decisions came back.
"""

from langchain_core.messages import SystemMessage
from langgraph.types import Send

from argus.llm import get_llm, get_token_usage
from argus.schemas import AgentDecision
from argus.state import State

SUMMARY_SYSTEM_PROMPT = """Summarize the outcome of this conversation as \
a structured decision. Base it ONLY on what was actually concluded above \
-- including any tool results already present -- not on the customer's \
original request alone."""


def build_summarize_node(agent_name: str, real_tool_names: list[str]):
    """Factory: one call produces one node function, closed over
    agent_name. Same reuse-not-duplicate shape as harness.build_tools_node.

    real_tool_names: the specialist's ACTUAL data/decision tools (e.g.
    ["get_fraud_score"]) -- deliberately NOT including handoff-signal
    tools like claims_triage.py's flag_for_fraud_review. See the note
    below on why this distinction is load-bearing, not pedantic.
    """
    structured_llm = get_llm().with_structured_output(AgentDecision, include_raw=True)

    def summarize_decision(state: State) -> dict:
        # A "no tool_calls in the LAST message" turn covers TWO different
        # situations: a real final answer following a completed REAL tool
        # call, AND an agent that's still asking a clarifying question and
        # has never called one of ITS OWN real tools. Caught live: a naive
        # "any ToolMessage in history" check still gets fooled by Claims'
        # handoff ack (also a ToolMessage, but for flag_for_fraud_review,
        # not a real data tool) -- Fraud would see that ack already in
        # history and wrongly think a real result existed. Checking the
        # tool NAME against this specialist's own real tools, not just
        # "is there a ToolMessage at all", is what actually fixes it.
        has_real_tool_result = any(
            getattr(m, "type", None) == "tool" and getattr(m, "name", None) in real_tool_names
            for m in state["messages"]
        )
        if not has_real_tool_result:
            print(
                f"[compliance] {agent_name}: no tool was called yet -- "
                "skipping decision summary, nothing real to audit"
            )
            return {"decisions": []}

        # state["messages"] at this point ends with the specialist's own
        # final AI answer -- a model-turn message. Gemini rejects a
        # request that ends in a model turn with nothing to respond to
        # ("Requests ending with a model turn are not supported.", hit
        # live). Append an explicit trailing human turn so there's
        # something to actually respond to.
        messages = (
            [SystemMessage(SUMMARY_SYSTEM_PROMPT)]
            + state["messages"]
            + [("user", "Summarize the decision above as a structured AgentDecision.")]
        )
        response = structured_llm.invoke(messages)
        result: AgentDecision = response["parsed"]
        tokens = get_token_usage(response["raw"])
        # Don't trust the model for something the calling code already
        # knows deterministically -- we KNOW which agent this is, it's a
        # parameter, not something to ask the LLM to get right.
        result.agent = agent_name
        print(
            f"[compliance] {agent_name} decision={result.decision!r} "
            f"confidence={result.confidence} reasons={result.reason_codes} tokens={tokens}"
        )
        return {"decisions": [result.model_dump()], "total_tokens_used": tokens}

    return summarize_decision


def dispatch_to_compliance(state: State) -> list[Send]:
    """Lives in the PARENT graph. Fans out over state['decisions'] --
    usually 0 or 1 entries today (one specialist completes per request;
    Claims' handoff means Claims itself never reaches summarize_decision,
    only Fraud does), but genuinely supports more.
    """
    return [
        Send("log_decision", {"decisions": [d], "audit_log": []})
        for d in state["decisions"]
    ]


def log_decision(state: State) -> dict:
    d = state["decisions"][0]
    entry = (
        f"[AUDIT] agent={d['agent']} decision={d['decision']} "
        f"confidence={d['confidence']} reasons={d['reason_codes']}"
    )
    return {"audit_log": [entry]}
