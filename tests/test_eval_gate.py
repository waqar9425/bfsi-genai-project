"""
The one thing that matters most about a CI gate: it must actually FAIL
when something's wrong, not just always report PASS. Tested here by
forcing a failure into the offline evals (monkeypatched, no LLM needed)
and confirming main() returns a non-zero exit code -- a gate that can
only ever return 0 is worse than no gate at all, a false sense of safety.
"""

from argus import eval_gate


def test_report_reflects_the_passed_in_result():
    assert eval_gate._report("some check", True) is True
    assert eval_gate._report("some check", False) is False


def test_main_returns_zero_when_everything_passes(monkeypatch):
    monkeypatch.setattr(eval_gate, "run_offline_evals", lambda: {"skill": [("case", True)]})
    monkeypatch.setattr(eval_gate, "context_precision", lambda: (1.0, []))
    monkeypatch.setattr(eval_gate.asyncio, "run", lambda coro: True)

    assert eval_gate.main() == 0


def test_main_returns_nonzero_when_an_offline_eval_fails(monkeypatch):
    # A single FAILING offline case must be enough to fail the whole gate,
    # even if everything downstream would have passed.
    monkeypatch.setattr(
        eval_gate, "run_offline_evals", lambda: {"skill": [("a real regression", False)]}
    )
    monkeypatch.setattr(eval_gate, "context_precision", lambda: (1.0, []))
    monkeypatch.setattr(eval_gate.asyncio, "run", lambda coro: True)

    assert eval_gate.main() == 1


def test_main_returns_nonzero_when_context_precision_is_below_threshold(monkeypatch):
    monkeypatch.setattr(eval_gate, "run_offline_evals", lambda: {"skill": [("case", True)]})
    monkeypatch.setattr(eval_gate, "context_precision", lambda: (0.5, [{"hit": False}]))
    monkeypatch.setattr(eval_gate.asyncio, "run", lambda coro: True)

    assert eval_gate.main() == 1


def test_main_returns_nonzero_when_live_checks_fail(monkeypatch):
    monkeypatch.setattr(eval_gate, "run_offline_evals", lambda: {"skill": [("case", True)]})
    monkeypatch.setattr(eval_gate, "context_precision", lambda: (1.0, []))
    monkeypatch.setattr(eval_gate.asyncio, "run", lambda coro: False)  # simulates a live-eval failure

    assert eval_gate.main() == 1
