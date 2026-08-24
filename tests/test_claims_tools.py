import pytest

from argus.tools.claims_tools import flag_for_fraud_review, get_claim_severity


def test_known_claim_type_returns_severity():
    result = get_claim_severity.invoke({"claim_type": "fire"})
    assert result["severity"] == "severe"
    assert result["avg_cost"] == 45000


def test_claim_type_lookup_is_case_and_whitespace_insensitive():
    result = get_claim_severity.invoke({"claim_type": "  Water_Damage "})
    assert result["claim_type"] == "water_damage"


def test_unknown_claim_type_raises():
    with pytest.raises(ValueError, match="Unknown claim_type"):
        get_claim_severity.invoke({"claim_type": "asteroid_impact"})


def test_flag_for_fraud_review_returns_ack_payload():
    result = flag_for_fraud_review.invoke({"reason": "duplicate claim details"})
    assert result["flagged"] is True
    assert result["reason"] == "duplicate claim details"
