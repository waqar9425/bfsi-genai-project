"""
Milestone 12: the Argus MCP server -- the real tool transport layer the
blueprint's architecture always specified ("all tool access... goes
through MCP as typed contracts"), deferred since Milestone 2 until the
core agent loop was proven first.

This WRAPS the existing tools/*.py functions rather than reimplementing
them -- the mock logic itself doesn't change AT ALL, only how a
specialist REACHES it changes. This is the blueprint's "swap the mock for
a real trained model without touching agent code" premise made literal:
that swap will happen entirely inside tools/*.py (or wherever this file
points instead), and every specialist stays exactly as it is today.

Existing LangChain @tool decorators on tools/*.py are left in place on
purpose -- test_fraud_tools.py etc. still import and test them directly,
zero-LLM-cost, unaffected by any of this.

Run standalone to sanity-check the server directly:
    python -m argus.mcp_server
Real usage is via mcp_client.py, which launches this as a subprocess over
stdio -- nothing talks to this file directly except through that.
"""

import sys
from pathlib import Path

# This file is launched as a SUBPROCESS by the MCP client (mcp_client.py),
# which does not necessarily inherit the parent process's PYTHONPATH --
# verified directly: it doesn't, launching failed with "No module named
# 'argus'" until this was added. Make the server self-sufficient about its
# own importability rather than depending on however it happens to be
# launched (relevant again once this runs under Docker/CI later).
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from argus.tools.claims_tools import flag_for_fraud_review as _flag_for_fraud_review  # noqa: E402
from argus.tools.claims_tools import get_claim_severity as _get_claim_severity
from argus.tools.fraud_tools import get_fraud_score as _get_fraud_score
from argus.tools.policy_tools import search_policy_docs as _search_policy_docs
from argus.tools.risk_tools import get_risk_grade as _get_risk_grade

mcp = FastMCP("argus-tools")


@mcp.tool()
def get_fraud_score(claim_amount: float, prior_claims_count: int, claim_hour: int) -> dict:
    """Score an insurance claim for fraud risk given its amount, the
    claimant's prior claims count, and the hour of day (0-23) filed.
    """
    return _get_fraud_score.invoke(
        {"claim_amount": claim_amount, "prior_claims_count": prior_claims_count, "claim_hour": claim_hour}
    )


@mcp.tool()
def get_risk_grade(loan_amount: float, annual_income: float) -> dict:
    """Grade credit/underwriting risk from a loan amount and annual income."""
    return _get_risk_grade.invoke({"loan_amount": loan_amount, "annual_income": annual_income})


@mcp.tool()
def get_claim_severity(claim_type: str) -> dict:
    """Look up the severity band and average cost for a claim type."""
    return _get_claim_severity.invoke({"claim_type": claim_type})


@mcp.tool()
def flag_for_fraud_review(reason: str) -> dict:
    """Flag a claim for fraud investigation instead of continuing normal
    claims triage. NOTE: claims_triage.py intercepts a call to this
    BEFORE it ever reaches this server (see call_model there) -- it's
    registered here purely so the model can discover/be offered it as an
    option at all. Real invocation of this specific tool never happens.
    """
    return _flag_for_fraud_review.invoke({"reason": reason})


@mcp.tool()
def search_policy_docs(query: str) -> list[dict]:
    """Search policy wordings, coverage rules, claims process, and KYC
    documentation for information relevant to a customer's question.
    """
    return _search_policy_docs.invoke({"query": query})


if __name__ == "__main__":
    mcp.run(transport="stdio")
