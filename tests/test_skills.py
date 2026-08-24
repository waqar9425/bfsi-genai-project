"""
Proves the "sharing" claim is real, not just visually similar prompts --
Fraud and Underwriting must contain the LITERAL SAME instructions text
from the same Skill object.

human_escalation's own ESCALATION_SKILL usage is tested in
test_checkpointing.py instead, not here -- since Milestone 11, it calls
interrupt() and can no longer be invoked as a bare function outside a
compiled graph (see that file for why).
"""

from argus.agents.fraud_investigation import FRAUD_AGENT_SYSTEM_PROMPT
from argus.agents.underwriting_risk import UNDERWRITING_AGENT_SYSTEM_PROMPT
from argus.skills import REASON_CODE_SKILL


def test_fraud_and_underwriting_share_the_exact_same_reason_code_text():
    assert REASON_CODE_SKILL.instructions in FRAUD_AGENT_SYSTEM_PROMPT
    assert REASON_CODE_SKILL.instructions in UNDERWRITING_AGENT_SYSTEM_PROMPT
