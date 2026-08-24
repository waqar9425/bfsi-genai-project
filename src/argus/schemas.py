"""
Pydantic schemas for structured LLM outputs.

Every agent response that needs to be routed on, validated, or logged to
the compliance trail goes through a schema like this -- never free text.
This is what lets the router and the audit log consume agent output
reliably instead of regex-scraping a paragraph the model wrote.
"""

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["fraud", "claims", "underwriting", "policy"]


class IntentClassification(BaseModel):
    """The Orchestrator's routing decision for one user turn."""

    intent: Intent = Field(
        description=(
            "'fraud' for suspicious transactions or fraud reports. "
            "'claims' for filing or checking an insurance claim. "
            "'underwriting' for policy pricing, loan or credit risk questions. "
            "'policy' for coverage questions, KYC, or general account queries."
        )
    )
    reasoning: str = Field(
        description="One short sentence explaining why this intent was chosen."
    )


class AgentDecision(BaseModel):
    """What a specialist actually decided, in typed form -- produced by
    build_summarize_node() (compliance.py) after a specialist's ReAct loop
    finishes normally. This is the object the Compliance & Audit fan-out
    (Milestone 8) logs; nothing downstream should ever have to parse the
    specialist's free-text reply to figure out what it decided.
    """

    agent: str = Field(description="Which specialist made this decision, e.g. 'fraud'")
    decision: str = Field(
        description="A short label for the outcome, e.g. 'high_risk_flagged', 'grade_B', 'triaged_minor'"
    )
    confidence: float = Field(description="0.0 to 1.0 -- how confident the agent is in this decision")
    reason_codes: list[str] = Field(
        description="Short, plain-language reasons supporting the decision"
    )
