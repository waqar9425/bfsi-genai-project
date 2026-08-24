"""
Milestone 9, Part A: PII/KYC redaction.

A guardrail applied at the graph's entry point (see graph.py -- this runs
BEFORE classify_intent), so no raw PII ever reaches an LLM call, gets
logged to the audit trail, or gets echoed back in a response. Regex-based
-- a real production system would likely use a dedicated PII-detection
model or a maintained library (e.g. Microsoft Presidio) for better
recall and fewer false positives/negatives; this is the mocked,
learning-project version of the same idea, same "mocks before
infrastructure" philosophy as the classical ML tools.

Known limitations, named not hidden:
- SSN pattern requires dashes (123-45-6789), misses 123456789.
- Credit card pattern matches common 4-4-4-4 formats only, and does NOT
  validate via the Luhn checksum -- a real system would, to cut down
  false positives on any random 16-digit number.
- Phone pattern is US-format-shaped only.
"""

import re

from langchain_core.messages import HumanMessage

from argus.state import State

_PATTERNS = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
}


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Pure function -- no LLM, no graph. Returns (redacted_text, types_found).
    Order matters: SSN and PHONE patterns can overlap on some inputs, so
    both are applied regardless of what the other already matched.
    """
    found = []
    for label, pattern in _PATTERNS.items():
        if pattern.search(text):
            found.append(label)
            text = pattern.sub(f"[REDACTED_{label}]", text)
    return text, found


def redact_pii_node(state: State) -> dict:
    """Graph node wrapping redact_pii. Edits the last message IN PLACE by
    returning a replacement with the SAME message id -- add_messages'
    reducer upserts by id instead of appending a duplicate (verified
    empirically before relying on this).

    Milestone 11: this is also the node that runs first on EVERY turn
    (including turn 2, 3, ... of a persisted conversation -- see
    graph.py), which makes it the natural place to reset `agent_turns` to
    0 every turn -- see state.py for why that reset has to happen
    explicitly now, rather than relying on a reducer.
    """
    update = {"agent_turns": 0}

    last = state["messages"][-1]
    if getattr(last, "type", None) != "human":
        return update

    redacted_content, found = redact_pii(last.content)
    if not found:
        return update

    print(f"[guardrails] redacted PII types in user message: {found}")
    update["messages"] = [HumanMessage(content=redacted_content, id=last.id)]
    return update
