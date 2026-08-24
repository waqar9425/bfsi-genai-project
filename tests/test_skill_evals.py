"""
Offline (fixture-based) skill evals -- pure Python, zero LLM calls. Proves
the RUBRIC CHECK FUNCTIONS correctly tell good output from bad, which has
to be true before trusting them to gate anything real (live evals, or
eventually CI). Live evals against genuine model output are a manual run
(python -m argus.skill_evals), not part of this suite -- consistent with
every other LLM-dependent piece in this project.
"""

from argus.skill_evals import run_offline_evals


def test_all_offline_eval_cases_match_expected_pass_fail():
    results = run_offline_evals()
    for skill_name, cases in results.items():
        for description, matched_expectation in cases:
            assert matched_expectation, f"{skill_name}: {description!r} did not match expected pass/fail"


def test_offline_evals_cover_all_three_llm_skills():
    results = run_offline_evals()
    assert set(results.keys()) == {"reason_code", "fraud_narrative", "coverage_lookup"}
