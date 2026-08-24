"""
Mocked claims-severity tool -- Phase 1 stand-in per the blueprint's
Section 07 spec: "Category-average lookup table." Real model (Gradient
Boosting Regressor) replaces this later behind the same contract.

Also holds `flag_for_fraud_review` -- not a data-lookup tool at all, but a
DECISION the Claims Triage agent can make: "stop normal triage, hand this
off to Fraud Investigation." See claims_triage.py for how its call gets
intercepted and turned into a Command handoff rather than executed as a
normal tool.
"""

from langchain_core.tools import tool

_SEVERITY_TABLE = {
    "auto_collision": {"severity": "moderate", "avg_cost": 8500},
    "water_damage": {"severity": "moderate", "avg_cost": 6200},
    "theft": {"severity": "minor", "avg_cost": 2100},
    "fire": {"severity": "severe", "avg_cost": 45000},
    "total_loss": {"severity": "severe", "avg_cost": 28000},
}


@tool
def get_claim_severity(claim_type: str) -> dict:
    """Look up the severity band and average cost for a claim type.

    Args:
        claim_type: one of 'auto_collision', 'water_damage', 'theft',
            'fire', 'total_loss'.

    Use this whenever you need an actual severity assessment -- never
    estimate one yourself from the conversation.
    """
    key = claim_type.lower().strip()
    entry = _SEVERITY_TABLE.get(key)
    if entry is None:
        raise ValueError(
            f"Unknown claim_type: {claim_type!r}. Must be one of {list(_SEVERITY_TABLE)}"
        )
    return {"claim_type": key, **entry}


@tool
def flag_for_fraud_review(reason: str) -> dict:
    """Flag this claim for fraud investigation instead of continuing
    normal claims triage. Call this if the claimant's account contains red
    flags -- inconsistent details, suspicious timing, an implausible
    story -- rather than trying to assess fraud yourself.

    Args:
        reason: A short explanation of what triggered the flag.
    """
    return {"flagged": True, "reason": reason}
