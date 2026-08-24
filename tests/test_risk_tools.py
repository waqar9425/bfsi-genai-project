from argus.tools.risk_tools import get_risk_grade


def _grade(**kwargs):
    return get_risk_grade.invoke(kwargs)


def test_grade_a_low_ratio():
    result = _grade(loan_amount=80_000, annual_income=45_000)  # ratio 1.78
    assert result["risk_grade"] == "A"


def test_grade_d_high_ratio():
    result = _grade(loan_amount=300_000, annual_income=40_000)  # ratio 7.5
    assert result["risk_grade"] == "D"


def test_boundary_ratio_is_inclusive():
    # ratio exactly 2.0 should land in grade A ("<= 2"), not spill into B
    result = _grade(loan_amount=90_000, annual_income=45_000)
    assert result["loan_to_income_ratio"] == 2.0
    assert result["risk_grade"] == "A"


def test_zero_income_raises_instead_of_dividing_by_zero():
    import pytest

    with pytest.raises(ValueError, match="annual_income"):
        get_risk_grade.invoke({"loan_amount": 50_000, "annual_income": 0})
