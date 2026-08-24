"""
Mocked credit/policy risk-grading tool -- Phase 1 stand-in per the
blueprint's Section 07 spec: "Bucket rule on loan-to-income ratio."
Real model (Logistic Regression baseline -> XGBoost challenger) replaces
this later behind the same contract, same as fraud_tools.py.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class RiskGradeResponse(BaseModel):
    """Typed contract every caller gets back -- mock or real."""

    risk_grade: str = Field(description="'A' (lowest risk) through 'D' (highest risk)")
    loan_to_income_ratio: float = Field(description="loan_amount / annual_income")
    rationale: str = Field(description="Plain-language explanation of the grade")


@tool
def get_risk_grade(loan_amount: float, annual_income: float) -> dict:
    """Grade credit/underwriting risk from a loan amount and the
    applicant's annual income.

    Args:
        loan_amount: The requested loan or policy coverage amount.
        annual_income: The applicant's stated annual income.

    Use this whenever you need an actual risk grade -- never estimate one
    yourself from the conversation.
    """
    if annual_income <= 0:
        # A real bug class, not a contrived one: bad/missing income data
        # is common in real intake. Fail loudly and typed, don't divide
        # by zero and don't silently default to some grade.
        raise ValueError("annual_income must be greater than 0")

    ratio = loan_amount / annual_income

    if ratio <= 2:
        grade, rationale = "A", "Loan-to-income ratio is well within a safe range."
    elif ratio <= 4:
        grade, rationale = "B", "Loan-to-income ratio is moderate."
    elif ratio <= 6:
        grade, rationale = "C", "Loan-to-income ratio is elevated."
    else:
        grade, rationale = "D", "Loan-to-income ratio is high relative to income."

    response = RiskGradeResponse(
        risk_grade=grade, loan_to_income_ratio=round(ratio, 2), rationale=rationale
    )
    return response.model_dump()
