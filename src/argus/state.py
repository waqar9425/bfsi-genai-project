"""
Argus shared state schema.

This is the object that flows through every node in the orchestrator graph.
Every node receives the current State and returns a *partial* dict of the
keys it wants to update -- LangGraph merges that dict back into State using
each field's reducer (see the `messages` field below for why that matters).
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class State(TypedDict):
    # `add_messages` is a *reducer*: instead of the new value replacing the
    # old one, LangGraph appends to the existing list. Without this
    # annotation, every node that touches `messages` would silently wipe
    # conversation history because dict.update() overwrites by default.
    messages: Annotated[list, add_messages]

    # Plain fields (no Annotated reducer) -> last-write-wins. A node that
    # returns {"intent": "fraud"} simply overwrites whatever was there.
    intent: str

    # Set by the harness (see harness.py) when a tool fails after its
    # retry budget is exhausted. Read by the post-tools router to decide
    # whether to loop back to the agent or divert to human escalation.
    needs_escalation: bool

    # Milestone 8: reducer-backed (operator.add = list concatenation) --
    # usually holds 0 or 1 entries today (one specialist completes per
    # request), but genuinely supports more than one, and is fanned out
    # over via Send in compliance.py (Milestone 7's pattern, applied for
    # real). Each entry is an AgentDecision.model_dump() dict, not the
    # Pydantic object itself -- state should hold plain, serializable data.
    decisions: Annotated[list[dict], operator.add]

    # Reducer-backed for the same reason as `decisions` -- N parallel
    # log_decision branches (Milestone 7's fanout.py pattern) each append
    # their own entry; without the reducer this would crash exactly like
    # the Milestone 0 concurrent-write experiment.
    audit_log: Annotated[list[str], operator.add]

    # Milestone 9, corrected in Milestone 11: originally an operator.add
    # reducer. Checkpointing (Milestone 11) exposed why that was wrong --
    # verified live: it kept accumulating ACROSS separate persisted turns
    # (turn 1 ended at 1, turn 2 ended at 3, not reset to a fresh count),
    # causing premature escalation later in a long conversation for no
    # real reason. `decisions`/`audit_log` above are CORRECTLY
    # thread-lifetime-scoped (a growing audit trail is exactly what you
    # want) -- agent_turns is different: it's meant to bound ONE turn's
    # tool-calling loop, not the whole conversation. Not every
    # reducer-backed field has the same correct scope; each one has to be
    # reasoned about on its own terms. Now plain (no reducer, last-write-
    # wins), explicitly reset to 0 every turn in guardrails.redact_pii_node
    # (the first node every turn hits), and read-then-incremented by
    # harness.build_tools_node instead of relying on auto-accumulation.
    agent_turns: int

    # Still reducer-backed, and DELIBERATELY so, unlike agent_turns above --
    # total cost of the WHOLE conversation is the useful metric here, so
    # accumulating across every persisted turn on a thread is correct, not
    # a bug. Reported by every LLM-calling node from
    # response.usage_metadata["total_tokens"] -- tracked/logged, not yet
    # enforced as a hard cap (see LEARNING_NOTES.md for why that's a
    # deliberate scope cut, not an oversight).
    total_tokens_used: Annotated[int, operator.add]
