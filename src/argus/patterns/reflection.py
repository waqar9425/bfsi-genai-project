"""
Milestone 7, Part 2: a reflection (generate -> critique -> revise) loop.

Every tool-calling loop so far (Milestone 2 onward) is generate-ACT-observe:
the model acts on the WORLD (calls a tool) and reacts to what comes back.
This is a different loop shape: generate-CRITIQUE-revise -- the model acts
on its OWN prior output, judged against explicit criteria, with an
iteration budget bounding how many passes it gets. Same underlying lesson
as the harness's MAX_ATTEMPTS (Milestone 4), generalized from "retry one
tool call" to "retry a whole generate/critique cycle": a loop without an
explicit, enforced budget is a bug waiting to happen, model loop or not.
"""

from typing import TypedDict

from pydantic import BaseModel, Field

from langgraph.graph import END, START, StateGraph

from argus.llm import get_llm

MAX_REFLECTION_ATTEMPTS = 2  # total critique passes allowed before finalizing best-effort


class CritiqueResult(BaseModel):
    approved: bool = Field(description="True if the draft meets both criteria")
    feedback: str = Field(description="If not approved, specific, actionable feedback for revision")


class ReflectionState(TypedDict):
    task: str
    draft: str
    feedback: str
    attempts: int
    approved: bool


_llm = get_llm()
_critique_llm = _llm.with_structured_output(CritiqueResult)

GENERATE_SYSTEM_PROMPT = """You write short, clear claim-denial \
explanations for Argus, a BFSI platform. Write ONE paragraph, 2-3 \
sentences, stating the denial reason plainly."""

CRITIQUE_SYSTEM_PROMPT = """You review claim-denial explanations against \
exactly two criteria: (1) it clearly states the specific reason for \
denial, not vague language, and (2) it does not use presumptuous or \
accusatory language toward the claimant (e.g. assuming bad faith or \
fraud without evidence). Approve only if BOTH are met."""


def generate(state: ReflectionState) -> dict:
    prompt = f"Task: {state['task']}"
    if state.get("feedback"):
        prompt += (
            f"\n\nYour previous draft was rejected. Feedback: {state['feedback']}\n"
            "Revise accordingly."
        )
    response = _llm.invoke([("system", GENERATE_SYSTEM_PROMPT), ("user", prompt)])
    print(f"[generate] attempt {state.get('attempts', 0) + 1} draft: {response.content!r}")
    return {"draft": response.content}


def critique(state: ReflectionState) -> dict:
    attempts = state.get("attempts", 0) + 1
    result: CritiqueResult = _critique_llm.invoke(
        [
            ("system", CRITIQUE_SYSTEM_PROMPT),
            ("user", f"Review this draft:\n\n{state['draft']}"),
        ]
    )
    print(f"[critique] attempt {attempts}: approved={result.approved} feedback={result.feedback!r}")
    return {"approved": result.approved, "feedback": result.feedback, "attempts": attempts}


def route_after_critique(state: ReflectionState) -> str:
    if state["approved"]:
        return END
    if state["attempts"] >= MAX_REFLECTION_ATTEMPTS:
        print(
            f"[route_after_critique] budget exhausted "
            f"({state['attempts']}/{MAX_REFLECTION_ATTEMPTS}) -- finalizing best-effort draft"
        )
        return END
    return "generate"


def build_reflection_demo():
    g = StateGraph(ReflectionState)
    g.add_node("generate", generate)
    g.add_node("critique", critique)
    g.add_edge(START, "generate")
    g.add_edge("generate", "critique")
    g.add_conditional_edges("critique", route_after_critique)
    return g.compile()


if __name__ == "__main__":
    app = build_reflection_demo()
    result = app.invoke(
        {
            "task": (
                "Deny a claim because the reported damage occurred 3 weeks "
                "before the policy's effective start date."
            ),
            "draft": "",
            "feedback": "",
            "attempts": 0,
            "approved": False,
        }
    )
    print()
    print("FINAL DRAFT:", result["draft"])
    print("APPROVED:", result["approved"], "ATTEMPTS:", result["attempts"])
