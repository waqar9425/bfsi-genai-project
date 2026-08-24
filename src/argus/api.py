"""
Milestone 13: FastAPI service wrapper around the Argus graph.

Sits on top of the Milestone-12 MCP tool layer and the Milestone-11
checkpointer -- a real HTTP boundary around what's been a Python-only
system until now.

Design question flagged since Milestone 12 (mcp_client.py's docstring):
does asyncio.run()-based tool bootstrapping (used at each specialist's
IMPORT time) survive being imported by an async web framework? Verified
directly by running this as a real uvicorn process, not assumed: YES, as
long as the import happens at TRUE top-level module scope -- `from
argus.graph import build_graph` below runs synchronously during process
startup, BEFORE uvicorn's event loop exists, exactly like a plain script
or pytest collection. The failure mode the docstring worried about is
real, but only if this same import were deferred into an async
lifespan/startup hook instead -- verified that specifically breaks, with
exactly the "cannot run event loop while another is running" error
predicted. Import your heavy stuff at real top level; don't defer it into
`async def startup()`.
"""

import uuid

from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field

from argus.graph import build_graph
from argus.skill_evals import extract_text  # reused from Milestone 10, not reimplemented

# Built at TRUE top-level, synchronously, during import -- see module
# docstring. This is also the "compile once, reuse every request" pattern
# from Milestone 1, now serving actual HTTP requests instead of test runs.
_app_graph = build_graph()

app = FastAPI(title="Argus", description="BFSI multi-agent platform")


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = Field(default=None, description="Omit to start a new conversation")


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    paused: bool = False
    interrupt_payload: dict | None = None
    agent_turns: int = 0
    total_tokens_used: int = 0
    audit_log: list[str] = []


class ResumeRequest(BaseModel):
    thread_id: str
    decision: str


def _build_response(thread_id: str, result: dict) -> ChatResponse:
    """Shared by /chat and /resume -- both can end in either a normal
    reply or a fresh pause (nothing stops a resumed request from hitting
    ANOTHER escalation later in the same turn).
    """
    pending = result.get("__interrupt__")
    if pending:
        return ChatResponse(
            thread_id=thread_id,
            reply="This request needs human review before it can continue.",
            paused=True,
            interrupt_payload=pending[0].value,
            agent_turns=result.get("agent_turns", 0),
            total_tokens_used=result.get("total_tokens_used", 0),
            audit_log=result.get("audit_log", []),
        )
    last_message = result["messages"][-1]
    return ChatResponse(
        thread_id=thread_id,
        reply=extract_text(last_message.content),
        paused=False,
        agent_turns=result.get("agent_turns", 0),
        total_tokens_used=result.get("total_tokens_used", 0),
        audit_log=result.get("audit_log", []),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = await _app_graph.ainvoke(
        {"messages": [("user", req.message)], "intent": "", "needs_escalation": False},
        config=config,
    )
    return _build_response(thread_id, result)


@app.post("/resume", response_model=ChatResponse)
async def resume(req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    state = await _app_graph.aget_state(config)
    if not state.next:  # empty tuple = nothing paused on this thread -- verified directly
        raise HTTPException(status_code=400, detail="No paused request on this thread_id")
    result = await _app_graph.ainvoke(Command(resume=req.decision), config=config)
    return _build_response(req.thread_id, result)
