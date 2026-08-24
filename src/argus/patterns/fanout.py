"""
Milestone 7, Part 1: Send-based dynamic parallel fan-out.

A conditional-edge function can return a list of Send(node_name, state)
objects instead of a single node-name string -- each Send is an
independent, parallel invocation of that node with its OWN local state,
not the parent's. This is LangGraph's "map" primitive: dispatch N parallel
branches from a single node, without hand-writing N separate edges.

This is the pattern Milestone 8 (Compliance & Audit) uses for real: every
specialist's decision fans out to a shared logging node, all in parallel,
without duplicating logging code in each specialist.

Critical fact, verified empirically before writing this file: fan-out is
ONLY safe because `audit_log` has a reducer (operator.add). Remove the
reducer and this crashes with the EXACT SAME InvalidUpdateError as
Milestone 0's concurrent-write experiment -- Send-based fan-out is that
same scenario, used deliberately instead of stumbled into by accident.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class FanoutState(TypedDict):
    decisions: list[dict]
    audit_log: Annotated[list[str], operator.add]  # reducer required -- see module docstring


def dispatch(state: FanoutState) -> list[Send]:
    """Conditional-edge function returning Send objects instead of a
    string. Each Send('log_one', {...}) is a separate parallel run of
    log_one, seeded with exactly one decision -- not the whole list.
    """
    return [Send("log_one", {"decisions": [d], "audit_log": []}) for d in state["decisions"]]


def log_one(state: FanoutState) -> dict:
    d = state["decisions"][0]
    return {"audit_log": [f"[AUDIT] {d['agent']}: {d['decision']} (confidence={d.get('confidence', 'n/a')})"]}


def build_fanout_demo():
    g = StateGraph(FanoutState)
    g.add_node("dispatch", lambda s: {})  # pass-through -- all the work happens in the conditional edge
    g.add_node("log_one", log_one)
    g.add_edge(START, "dispatch")
    g.add_conditional_edges("dispatch", dispatch)  # no mapping dict -- Send already names its target
    g.add_edge("log_one", END)
    return g.compile()


if __name__ == "__main__":
    app = build_fanout_demo()
    result = app.invoke(
        {
            "decisions": [
                {"agent": "fraud", "decision": "flag_for_review", "confidence": 0.91},
                {"agent": "underwriting", "decision": "grade_B", "confidence": 0.77},
                {"agent": "claims", "decision": "triaged_minor", "confidence": 0.95},
            ],
            "audit_log": [],
        }
    )
    for entry in result["audit_log"]:
        print(entry)
