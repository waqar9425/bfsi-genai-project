"""
Milestone 11: checkpointing + interrupt/resume mechanics. Pure LangGraph
API, no LLM calls needed -- everything here is testable with plain,
hand-built minimal graphs, same principle as every other pure-logic test
in this project.
"""

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from argus.harness import human_escalation
from argus.skills import ESCALATION_SKILL
from argus.state import State


def _build_escalation_only_graph():
    g = StateGraph(State)
    g.add_node("human_escalation", human_escalation)
    g.add_edge(START, "human_escalation")
    g.add_edge("human_escalation", END)
    return g.compile(checkpointer=MemorySaver())


def test_human_escalation_pauses_with_interrupt():
    app = _build_escalation_only_graph()
    config = {"configurable": {"thread_id": "test-pause"}}

    result = app.invoke(
        {"messages": [], "intent": "", "needs_escalation": True, "agent_turns": 0},
        config=config,
    )

    assert "__interrupt__" in result  # paused, did NOT run to completion
    assert result["__interrupt__"][0].value["needs_escalation"] is True


def test_human_escalation_resumes_and_uses_escalation_skill():
    app = _build_escalation_only_graph()
    config = {"configurable": {"thread_id": "test-resume"}}

    app.invoke(
        {"messages": [], "intent": "", "needs_escalation": True, "agent_turns": 0}, config=config
    )  # pauses
    result = app.invoke(Command(resume="approved, proceed manually"), config=config)  # resumes

    assert "__interrupt__" not in result  # actually completed this time
    final_message = result["messages"][-1]
    assert isinstance(final_message, AIMessage)
    assert ESCALATION_SKILL.instructions in final_message.content
    assert "approved, proceed manually" in final_message.content


def test_checkpointer_persists_across_separate_invokes_same_thread():
    def echo(state):
        return {}

    g = StateGraph(State)
    g.add_node("echo", echo)
    g.add_edge(START, "echo")
    g.add_edge("echo", END)
    app = g.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "test-memory"}}
    app.invoke({"messages": [("user", "first")], "intent": "", "needs_escalation": False}, config=config)
    result = app.invoke({"messages": [("user", "second")]}, config=config)  # only the NEW message

    assert len(result["messages"]) == 2  # both turns present -- proves persistence
    assert result["messages"][0].content == "first"
    assert result["messages"][1].content == "second"


def test_different_thread_ids_are_fully_isolated():
    def echo(state):
        return {}

    g = StateGraph(State)
    g.add_node("echo", echo)
    g.add_edge(START, "echo")
    g.add_edge("echo", END)
    app = g.compile(checkpointer=MemorySaver())

    app.invoke(
        {"messages": [("user", "thread A message")], "intent": "", "needs_escalation": False},
        config={"configurable": {"thread_id": "thread-A"}},
    )
    result_b = app.invoke(
        {"messages": [("user", "thread B message")], "intent": "", "needs_escalation": False},
        config={"configurable": {"thread_id": "thread-B"}},
    )

    assert len(result_b["messages"]) == 1  # thread B never saw thread A's message
