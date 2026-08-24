"""
Milestone 10: Agent Skills -- reusable, versioned prompt fragments,
extracted out of what used to be one long inline system-prompt string per
specialist (Milestones 2-9). Per blueprint Section 06: "a prompt
template... callable by more than one agent instead of copy-pasted into
each."

Each skill also owns its own eval set (skill_evals.py) -- a prompt change
that regresses a skill's eval score is meant to block deployment the same
way a model-accuracy regression would (the blueprint's own framing, now
literal: see skill_evals.py's rubric checks).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Skill:
    name: str
    version: str
    instructions: str


REASON_CODE_SKILL = Skill(
    name="reason_code",
    version="1.0",
    instructions=(
        "When explaining a computed score or grade: state the result "
        "plainly first, then list the SPECIFIC factors that drove it, in "
        "plain language, not jargon. Never add reasoning or caveats the "
        "tool result didn't actually provide."
    ),
)

# Used by fraud_investigation.py only, but still modeled as a Skill --
# the point of a Skill is being independently versioned and eval-tested,
# not necessarily having 2+ callers today.
FRAUD_NARRATIVE_SKILL = Skill(
    name="fraud_narrative",
    version="1.0",
    instructions=(
        "When a fraud score is medium or high risk, phrase the "
        "explanation as an SIU-ready investigator note: state the risk "
        "band, list the drivers as objective facts -- never assert the "
        "claimant is lying or accuse them directly -- and end with a "
        "clear recommendation such as 'recommend manual review'."
    ),
)

COVERAGE_LOOKUP_SKILL = Skill(
    name="coverage_lookup",
    version="1.0",
    instructions=(
        "When answering from search_policy_docs results, cite the "
        "doc_id(s) you drew from inline, in square brackets, e.g. "
        "[COV-WATER-01]. If the results don't actually answer the "
        "question, say so plainly -- never answer from general insurance "
        "knowledge that isn't present in the retrieved documents."
    ),
)

# Not an LLM prompt fragment -- this is the fixed phrasing harness.py's
# human_escalation uses directly. Modeled as a Skill anyway to keep the
# "shared, versioned, named unit" property explicit rather than leaving
# it as an unlabeled string buried in harness.py.
ESCALATION_SKILL = Skill(
    name="escalation",
    version="1.0",
    instructions=(
        "I wasn't able to complete this automatically after retrying -- "
        "escalating to a human for review. Someone will follow up shortly."
    ),
)
