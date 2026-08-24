"""
The tool's scoring logic is a pure function -- test it directly via
.invoke(), no LLM/graph involved. Same principle as testing route_by_intent
in Milestone 0: isolate the deterministic part from the model-dependent part.
"""

from argus.tools.fraud_tools import get_fraud_score


def _score(**kwargs):
    return get_fraud_score.invoke(kwargs)


def test_high_risk_claim():
    result = _score(claim_amount=15000, prior_claims_count=4, claim_hour=2)
    assert result["risk_band"] == "high"
    assert result["fraud_score"] == 1.0


def test_low_risk_claim():
    result = _score(claim_amount=500, prior_claims_count=0, claim_hour=14)
    assert result["risk_band"] == "low"
    assert result["fraud_score"] == 0.0
    assert "No significant risk factors" in result["top_drivers"][0]


def test_moderate_amount_alone_is_not_enough_for_medium_band():
    # $6,000 alone only scores 0.2 (the >5000 branch) -- below the 0.3
    # medium-band threshold. One weak signal alone stays "low"; it takes a
    # second factor (prior claims, odd hour) to cross into "medium". This
    # is a real property of the mock rule, not an accident -- pin it down.
    result = _score(claim_amount=6000, prior_claims_count=0, claim_hour=14)
    assert result["risk_band"] == "low"
    assert result["fraud_score"] == 0.2


def test_score_never_exceeds_one():
    result = _score(claim_amount=999999, prior_claims_count=99, claim_hour=3)
    assert result["fraud_score"] <= 1.0
