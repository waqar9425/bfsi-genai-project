"""
Milestone 14: the CI-ready eval gate.

Combines Milestone 10's skill live-evals and this milestone's RAG
live-evals into ONE script with a real pass/fail exit code --
`python -m argus.eval_gate; echo $?` is exactly what a CI pipeline step
would run to gate a merge.

Deliberately SEPARATE from pytest, on purpose, not an oversight: pytest
covers cheap, deterministic logic and should run on every single push
(seconds, zero LLM cost). This covers expensive, LLM-dependent QUALITY
regressions -- a prompt change that makes a skill worse, a retrieval
change that starts missing documents, an answer that stops being
grounded in what was retrieved. Meant to gate a merge or run nightly, not
fire on every keystroke. Two-tier CI, matching the two-tier cost of what
each tier actually checks.
"""

import asyncio
import sys

# Imported at TRUE top-level, before asyncio.run() is ever called below --
# this is not a style preference, it's required. Verified live: these
# specialist modules call get_mcp_tools() (mcp_client.py) at THEIR OWN
# import time, which does its own internal asyncio.run(). Deferring these
# imports into an async function (as the first version of this file did)
# means that inner asyncio.run() fires WHILE already inside the outer
# asyncio.run()'s event loop -- "asyncio.run() cannot be called from a
# running event loop", hit for real, not just theorized. Same lesson
# api.py already had to apply (Milestone 13); missed it here the first
# time, since it wasn't as obvious this file needed it too.
from argus.agents.fraud_investigation import build_fraud_agent
from argus.agents.policy_customer import build_policy_agent
from argus.agents.underwriting_risk import build_underwriting_agent
from argus.rag_evals import context_precision, judge_answer
from argus.skill_evals import (
    check_coverage_lookup,
    check_fraud_narrative,
    check_reason_code,
    extract_text,
    run_offline_evals,
)

CONTEXT_PRECISION_THRESHOLD = 0.85


def _report(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}{(' -- ' + detail) if detail else ''}")
    return ok


async def _run_live_checks() -> bool:
    all_ok = True

    print("\n=== Live skill evals (real model output through real specialists) ===")

    fraud_app = build_fraud_agent()
    result = await fraud_app.ainvoke(
        {
            "messages": [("user", "Check this claim for fraud: $18,000, 5 prior claims, filed at 3am.")],
            "intent": "",
            "needs_escalation": False,
        }
    )
    text = extract_text(result["messages"][-1].content)
    all_ok &= _report("fraud_narrative", check_fraud_narrative(text))

    uw_app = build_underwriting_agent()
    result = await uw_app.ainvoke(
        {
            "messages": [("user", "Risk grade for a $300,000 loan against $40,000 income?")],
            "intent": "",
            "needs_escalation": False,
        }
    )
    text = extract_text(result["messages"][-1].content)
    all_ok &= _report("reason_code", check_reason_code(text))

    policy_app = build_policy_agent()
    question = "Is water damage from a burst pipe covered?"
    result = await policy_app.ainvoke(
        {"messages": [("user", question)], "intent": "", "needs_escalation": False}
    )
    answer = extract_text(result["messages"][-1].content)
    all_ok &= _report("coverage_lookup", check_coverage_lookup(answer))

    print("\n=== RAG faithfulness + relevance (LLM-as-judge, separate model call) ===")
    retrieved_context = next(m.content for m in result["messages"] if m.type == "tool")
    judgment = judge_answer(question, retrieved_context, answer)
    all_ok &= _report("faithful", judgment.faithful, judgment.reasoning)
    all_ok &= _report("relevant", judgment.relevant, judgment.reasoning)

    return all_ok


def main() -> int:
    all_ok = True

    print("=== Offline skill fixture evals (sanity check on the rubrics themselves) ===")
    for skill, cases in run_offline_evals().items():
        for desc, ok in cases:
            all_ok &= _report(f"{skill}: {desc}", ok)

    print("\n=== RAG context precision (retrieval hit-rate@3, simplified RAGAS metric) ===")
    precision, details = context_precision()
    misses = [d for d in details if not d["hit"]]
    all_ok &= _report(
        f"context_precision ({precision:.2f} >= {CONTEXT_PRECISION_THRESHOLD})",
        precision >= CONTEXT_PRECISION_THRESHOLD,
        f"misses: {misses}" if misses else "",
    )

    all_ok &= asyncio.run(_run_live_checks())

    print()
    if all_ok:
        print("EVAL GATE: PASSED")
        return 0
    print("EVAL GATE: FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
