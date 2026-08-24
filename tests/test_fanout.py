"""
Pure Python, no LLM -- the fan-out mechanic itself is fully deterministic
and testable without a model, same principle as every router function
tested so far.
"""

from argus.patterns.fanout import build_fanout_demo


def test_fanout_processes_every_decision_exactly_once():
    app = build_fanout_demo()
    result = app.invoke(
        {
            "decisions": [
                {"agent": "fraud", "decision": "flag", "confidence": 0.9},
                {"agent": "claims", "decision": "triaged", "confidence": 0.8},
            ],
            "audit_log": [],
        }
    )
    assert len(result["audit_log"]) == 2
    assert any("fraud" in entry and "flag" in entry for entry in result["audit_log"])
    assert any("claims" in entry and "triaged" in entry for entry in result["audit_log"])


def test_fanout_with_empty_decisions_produces_empty_log():
    app = build_fanout_demo()
    result = app.invoke({"decisions": [], "audit_log": []})
    assert result["audit_log"] == []
