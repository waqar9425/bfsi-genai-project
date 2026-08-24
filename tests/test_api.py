"""
api.py's _build_response is the one piece of real logic in the FastAPI
layer -- pure function, no graph/HTTP server needed to test it. The full
request/response flow (including MCP bootstrap, real LLM calls) is
exercised as a live uvicorn run instead, same split as everywhere else in
this project. Importing argus.api triggers the full MCP tool bootstrap
for all four specialists (via argus.graph) at collection time -- this is
the one test file in the suite that pays that cost, unavoidably, since
there's no way to import _build_response without it.
"""

from langchain_core.messages import AIMessage

from argus.api import _build_response


def test_normal_completion_is_not_paused():
    result = {
        "messages": [AIMessage(content="Grade A.")],
        "agent_turns": 1,
        "total_tokens_used": 500,
        "audit_log": ["[AUDIT] agent=underwriting decision=grade_A"],
    }
    response = _build_response("thread-1", result)

    assert response.paused is False
    assert response.reply == "Grade A."
    assert response.interrupt_payload is None
    assert response.audit_log == ["[AUDIT] agent=underwriting decision=grade_A"]


def test_normal_completion_extracts_text_from_content_blocks():
    # Gemini's content-block shape (Milestone 2's gotcha) -- _build_response
    # must not leak this to API callers as raw internal structure.
    result = {
        "messages": [AIMessage(content=[{"type": "text", "text": "Grade B."}])],
        "agent_turns": 1,
        "total_tokens_used": 300,
        "audit_log": [],
    }
    response = _build_response("thread-2", result)

    assert response.reply == "Grade B."
    assert isinstance(response.reply, str)


def test_interrupted_result_reports_paused_with_payload():
    class FakeInterrupt:
        value = {"reason": "turn budget exhausted", "agent_turns": 3}

    result = {
        "messages": [],
        "__interrupt__": [FakeInterrupt()],
        "agent_turns": 0,
        "total_tokens_used": 100,
        "audit_log": [],
    }
    response = _build_response("thread-3", result)

    assert response.paused is True
    assert response.interrupt_payload == {"reason": "turn budget exhausted", "agent_turns": 3}
    assert "human review" in response.reply
