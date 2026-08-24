"""
Milestone 7, Part 3: recursion_limit as the last-resort circuit breaker.

Every graph, even one with well-designed loop-exit conditions, benefits
from this as a backstop -- it's what stands between a bug in your loop
logic and a graph that genuinely never terminates. Deliberately built a
graph with NO exit condition here to watch the backstop actually fire,
rather than just take on faith that it exists.

Contrast with the harness's MAX_ATTEMPTS (Milestone 4): that's a budget on
ONE tool call's retries, enforced by our own code, catching a specific
known failure mode (a flaky tool). recursion_limit is a blunt,
graph-wide backstop enforced by LangGraph itself, catching ANY runaway
loop regardless of cause -- a bug in a router, an LLM that keeps calling
tools forever, anything. Belt and suspenders: the harness budget should
never actually need recursion_limit's help, but recursion_limit is what
protects you the day the harness budget itself has a bug.
"""

from typing import TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph


class PingPongState(TypedDict):
    count: int


def ping(state: PingPongState) -> dict:
    return {"count": state["count"] + 1}


def pong(state: PingPongState) -> dict:
    return {"count": state["count"] + 1}


def build_infinite_loop_demo():
    """Deliberately broken: no exit condition. This graph would run
    forever without recursion_limit -- that's the point of the demo.
    """
    g = StateGraph(PingPongState)
    g.add_node("ping", ping)
    g.add_node("pong", pong)
    g.add_edge(START, "ping")
    g.add_edge("ping", "pong")
    g.add_edge("pong", "ping")  # no path to END, ever
    return g.compile()


if __name__ == "__main__":
    app = build_infinite_loop_demo()
    try:
        app.invoke({"count": 0}, config={"recursion_limit": 10})
        print("completed -- this should never print, the graph has no exit")
    except GraphRecursionError as e:
        print(f"Backstop fired as expected: {e}")
