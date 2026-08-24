"""
Router functions are plain functions of State -> str. Test them directly,
no app.invoke() / no LLM / no graph execution needed at all.
"""

from argus.graph import route_by_intent


def test_route_by_intent_fraud():
    assert route_by_intent({"messages": [], "intent": "fraud"}) == "fraud"


def test_route_by_intent_claims():
    assert route_by_intent({"messages": [], "intent": "claims"}) == "claims"


def test_route_by_intent_policy():
    assert route_by_intent({"messages": [], "intent": "policy"}) == "policy"


def test_route_by_intent_underwriting():
    assert route_by_intent({"messages": [], "intent": "underwriting"}) == "underwriting"
