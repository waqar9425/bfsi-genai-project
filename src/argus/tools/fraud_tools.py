"""
Mocked fraud-scoring tool -- Phase 1 stand-in per the blueprint's
mock-before-model plan (Section 07). Weighted rule, not a trained model.

The point of Milestone 2 isn't the rule's accuracy -- it's a real
tool-calling contract a specialist agent can call, with a response shape
a trained XGBoost + SHAP model can later fill in without any agent code
changing (same function signature, same response shape, per the blueprint's
"swap the mock, touch nothing else" design).
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class FraudScoreResponse(BaseModel):
    """Typed contract every caller of this tool gets back -- mock or real."""

    fraud_score: float = Field(description="0.0 (no risk) to 1.0 (high risk)")
    risk_band: str = Field(description="'low', 'medium', or 'high'")
    top_drivers: list[str] = Field(
        description="Plain-language factors behind the score, for the Reason-Code skill"
    )


@tool
def get_fraud_score(
    claim_amount: float,
    prior_claims_count: int,
    claim_hour: int,
) -> dict:
    """Score an insurance claim for fraud risk.

    Args:
        claim_amount: The dollar amount being claimed.
        prior_claims_count: How many prior claims this claimant has filed.
        claim_hour: Hour of day (0-23) the claim was filed.

    Use this whenever you need an actual fraud risk number -- never
    estimate a fraud score yourself from the conversation.
    """
    score = 0.0
    drivers: list[str] = []

    if claim_amount > 10000:
        score += 0.4
        drivers.append(f"Claim amount (${claim_amount:,.0f}) is unusually high")
    elif claim_amount > 5000:
        score += 0.2
        drivers.append(f"Claim amount (${claim_amount:,.0f}) is above average")

    if prior_claims_count >= 3:
        score += 0.35
        drivers.append(f"{prior_claims_count} prior claims on file")
    elif prior_claims_count >= 1:
        score += 0.1

    if claim_hour < 6 or claim_hour >= 23:
        score += 0.25
        drivers.append(f"Filed at an unusual hour ({claim_hour}:00)")

    score = min(score, 1.0)
    band = "high" if score >= 0.6 else "medium" if score >= 0.3 else "low"

    if not drivers:
        drivers.append("No significant risk factors identified")

    response = FraudScoreResponse(
        fraud_score=round(score, 2), risk_band=band, top_drivers=drivers
    )
    # Validate through the typed schema, then hand back a plain dict --
    # this is what actually becomes the ToolMessage content the model reads.
    return response.model_dump()
