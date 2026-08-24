"""
Eval sets for each skill in skills.py.

Two layers, deliberately kept separate (same split used throughout this
project -- pure logic in pytest, LLM-dependent stuff as a manual run):

1. OFFLINE evals (run_offline_evals): deterministic rubric-check functions
   scored against hand-written fixture outputs. Pure Python, zero LLM
   calls, safe for the pytest suite. Proves the RUBRIC ITSELF correctly
   tells good output from bad -- necessary before trusting it to gate
   anything real.
2. LIVE evals (this file's __main__): the same rubric functions, scored
   against genuine output from the real specialists. Proves the
   skill+rubric combination works on actual model behavior, not just
   examples I wrote by hand.
"""

import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class EvalCase:
    description: str
    text: str
    should_pass: bool


def extract_text(content) -> str:
    """Message content is sometimes a plain str, sometimes a list of
    content blocks (Gemini does this -- the Milestone 2 gotcha). Normalize
    either shape to plain text so rubric checks don't have to care.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


# --- reason_code -------------------------------------------------------

def check_reason_code(text: str) -> bool:
    """Rubric: explains a specific factor, isn't just a bare verdict."""
    words = text.split()
    explanatory_kw = ("because", "due to", "driven by", "factor", "ratio", "amount", "claims", "based on")
    return len(words) >= 8 and any(kw in text.lower() for kw in explanatory_kw)


REASON_CODE_EVAL_CASES = [
    EvalCase(
        description="good: states result and a specific factor",
        text="Grade B, driven by a loan-to-income ratio of 4.0, which is moderate.",
        should_pass=True,
    ),
    EvalCase(description="bad: bare verdict, no explanation", text="Grade B.", should_pass=False),
]


# --- fraud_narrative -----------------------------------------------------

def check_fraud_narrative(text: str) -> bool:
    """Rubric: recommends action, avoids presumptuous/accusatory language."""
    lower = text.lower()
    accusatory = ("is lying", "is guilty", "clearly fraud", "fraudulent claimant")
    has_recommendation = any(kw in lower for kw in ("recommend", "review", "investigat"))
    is_accusatory = any(kw in lower for kw in accusatory)
    return has_recommendation and not is_accusatory


FRAUD_NARRATIVE_EVAL_CASES = [
    EvalCase(
        description="good: objective drivers, clear recommendation",
        text=(
            "High risk band. Drivers: high claim amount, 4 prior claims, "
            "unusual filing hour. Recommend manual review."
        ),
        should_pass=True,
    ),
    EvalCase(
        description="bad: accusatory language",
        text="This claimant is clearly lying about the incident and is committing fraud.",
        should_pass=False,
    ),
]


# --- coverage_lookup -----------------------------------------------------

_DOC_ID_PATTERN = re.compile(r"\[[A-Z]+-[A-Z]+-\d+\]")


def check_coverage_lookup(text: str) -> bool:
    """Rubric: cites at least one doc_id in [BRACKET] format."""
    return bool(_DOC_ID_PATTERN.search(text))


COVERAGE_LOOKUP_EVAL_CASES = [
    EvalCase(
        description="good: cites a doc_id",
        text="Water damage from a burst pipe is covered [COV-WATER-01].",
        should_pass=True,
    ),
    EvalCase(
        description="bad: no citation",
        text="Water damage from a burst pipe is generally covered by most policies.",
        should_pass=False,
    ),
]


def run_offline_evals() -> dict[str, list[tuple[str, bool]]]:
    """Runs every skill's fixture cases through its rubric check. Returns
    {skill_name: [(case_description, matched_expected_pass_fail), ...]}.
    This is what the pytest suite exercises -- zero LLM calls.
    """
    suites: dict[str, tuple[Callable, list[EvalCase]]] = {
        "reason_code": (check_reason_code, REASON_CODE_EVAL_CASES),
        "fraud_narrative": (check_fraud_narrative, FRAUD_NARRATIVE_EVAL_CASES),
        "coverage_lookup": (check_coverage_lookup, COVERAGE_LOOKUP_EVAL_CASES),
    }
    results = {}
    for name, (check_fn, cases) in suites.items():
        results[name] = [(case.description, check_fn(case.text) == case.should_pass) for case in cases]
    return results


if __name__ == "__main__":
    import asyncio

    from argus.agents.fraud_investigation import build_fraud_agent
    from argus.agents.policy_customer import build_policy_agent
    from argus.agents.underwriting_risk import build_underwriting_agent

    async def main():
        print("=== Offline (fixture) evals ===")
        for skill, cases in run_offline_evals().items():
            for desc, ok in cases:
                print(f"  [{skill}] {'PASS' if ok else 'FAIL'} -- {desc}")

        print("\n=== Live evals (real model output through real specialists) ===")
        # Milestone 12: these specialists' graphs now require .ainvoke()
        # (MCP-backed tools node) -- see harness.py's module docstring.

        fraud_app = build_fraud_agent()
        result = await fraud_app.ainvoke(
            {
                "messages": [
                    ("user", "Check this claim for fraud: $18,000, 5 prior claims, filed at 3am.")
                ],
                "intent": "",
                "needs_escalation": False,
            }
        )
        text = extract_text(result["messages"][-1].content)
        print(f"  [fraud_narrative] {'PASS' if check_fraud_narrative(text) else 'FAIL'} -- live fraud agent output")
        print(f"    text: {text!r}")

        uw_app = build_underwriting_agent()
        result = await uw_app.ainvoke(
            {
                "messages": [("user", "Risk grade for a $300,000 loan against $40,000 income?")],
                "intent": "",
                "needs_escalation": False,
            }
        )
        text = extract_text(result["messages"][-1].content)
        print(f"  [reason_code] {'PASS' if check_reason_code(text) else 'FAIL'} -- live underwriting agent output")
        print(f"    text: {text!r}")

        policy_app = build_policy_agent()
        result = await policy_app.ainvoke(
            {
                "messages": [("user", "Is fire damage covered?")],
                "intent": "",
                "needs_escalation": False,
            }
        )
        text = extract_text(result["messages"][-1].content)
        print(f"  [coverage_lookup] {'PASS' if check_coverage_lookup(text) else 'FAIL'} -- live policy agent output")
        print(f"    text: {text!r}")

    asyncio.run(main())
