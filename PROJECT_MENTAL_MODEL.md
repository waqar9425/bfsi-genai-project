# Argus — Project Mental Model

The "what is this thing, actually" reference. Read this alongside
`LANGGRAPH_CONCEPTS.md` (the agent framework mechanics) and
`CONTAINERS_AND_CICD_CONCEPTS.md` (Docker/Podman/CI-CD mechanics) --
this file is the domain + architecture + "which file does what" side.

**Updated after every milestone.**

---

## 1. What Argus is, in one paragraph

Argus is a multi-agent system for a BFSI (banking/insurance) company. A
user sends one message (a claim report, a coverage question, a fraud tip,
a loan risk question). One **orchestrator** reads it and routes it to one
of four **specialist agents**, each an expert in one domain area.
Specialists never invent numbers -- when a decision needs an actual number
(a fraud score, a risk grade), the specialist calls a **tool** that
computes it deterministically, and just explains the result in plain
language. Every specialist's final decision gets logged to a **compliance
audit trail**. Right now the tools are simple rule-based mocks (a
deliberate, temporary choice -- see Section 5); they'll eventually be
swapped for real trained ML models without the agents' code changing at
all.

---

## 2. BFSI/insurance vocabulary used in this project

| Term | Plain meaning |
|---|---|
| **Policy** | The contract between the customer and the insurer -- what's covered, for how much, for how long. |
| **Premium** | What the customer pays (monthly/annually) to keep the policy active. |
| **Claim** | A request to actually get paid out after something bad happened (accident, theft, fire, etc.). |
| **Underwriting** | Deciding whether to offer coverage/a loan at all, and at what price/risk grade, *before* anything bad has happened. |
| **Deductible** | The amount the customer pays out of pocket before insurance covers the rest. |
| **Severity** (of a claim) | Roughly how bad/expensive the claim is likely to be -- minor/moderate/severe. |
| **Fraud investigation** | Checking whether a claim looks fake or exaggerated, using patterns (amount, timing, claim history) rather than trusting the story at face value. |
| **Risk grade** | A letter grade (A-D in this project) summarizing how risky it is to lend to or insure someone -- driven by things like loan-to-income ratio. |
| **KYC** (Know Your Customer) | Identity verification required before doing business with someone -- named in the blueprint, not yet built here. |
| **SIU** | Special Investigation Unit -- the real-world team a suspicious claim gets escalated to. Our `flag_for_fraud_review` handoff is a toy version of "sending it to SIU." |

---

## 3. The whole system, as a diagram

```
        HTTP POST /chat {message, thread_id?} or /resume {thread_id, decision}
                        (api.py, Milestone 13 -- FastAPI, real HTTP boundary)
                                │
                                ▼
        user message + a thread_id (Milestone 11 -- same thread_id
        across separate requests = same persisted conversation;
        different thread_id = fully isolated, fresh state)
                                │
                                ▼
                    ┌────────────────────┐
                    │     redact_pii       │  regex-based PII/KYC scrub,
                    │   (guardrails.py)     │  runs BEFORE any LLM call sees
                    └──────────┬────────────┘  the raw message (Milestone 9)
                                ▼
   ╔══════════════════ ORCHESTRATOR (graph.py) ═══════════════════╗
   ║   classify_intent -- ONE forced LLM call, decides:            ║
   ║   "fraud" | "claims" | "underwriting" | "policy"               ║
   ╚══════════════╤═══════════════╤═══════════════╤════════════════╝
                  ▼               ▼               ▼            ▼
             ┌────────┐     ┌─────────┐     ┌────────────┐  ┌────────┐
             │ fraud  │     │ claims  │     │underwriting│  │ policy │
             │(agent) │◀────│ can hand│     │  (agent)   │  │ (agent,│
             │        │hand-│ off HERE│     │            │  │  RAG)  │
             └───┬────┘ off └────┬────┘     └─────┬──────┘  └───┬────┘
                 │               │                 │             │
                 └───────────────┴────────┬────────┴─────────────┘
                                           ▼
                              ┌─────────────────────┐
                              │      compliance       │  fans out (Send)
                              │  (structured decision  │  over every
                              │   logged, if any real   │  decision that
                              │   decision was reached) │  actually happened
                              └──────────┬─────────────┘
                                         ▼
                                        END
```

Each of the four specialist boxes is internally the SAME shape (see
Section 4) -- what differs between them is only which tool(s) they call
and what their system prompt says.

**Since Milestone 12**, "calls a tool" for any specialist actually means:
```
   specialist's "tools" node (harness.py, now ASYNC)
                │
                │  .ainvoke(full_tool_call_dict) over stdio
                ▼
   ┌─────────────────────────────────────────┐
   │   Argus MCP server (mcp_server.py)         │  a SEPARATE subprocess,
   │   -- wraps tools/*.py's SAME functions      │  launched by mcp_client.py
   │   -- get_fraud_score, get_risk_grade,       │  at each specialist's
   │      get_claim_severity, search_policy_docs,│  own import time
   │      flag_for_fraud_review                  │
   └─────────────────────────────────────────┘
```
The mock logic in `tools/*.py` never changed -- only the transport did.
Real model swap (Section 6) will happen entirely inside this box.

---

## 4. The specialist "anatomy" -- one shape, reused four times

```
   ┌───────┐  tool_calls?  ┌────────────────┐   ok, loop back
   │ agent │──────────────▶│ tools (harness) │─────────────────┐
   └───┬───┘◀───────────────└────────┬────────┘                │
       │ no tool_calls               │ retry budget exhausted   │
       ▼                             ▼                          │
  ┌────────────┐            ┌─────────────────┐                 │
  │ summarize_ │            │ human_escalation │──▶ END          │
  │ decision   │            └─────────────────┘   (no decision   │
  └─────┬──────┘                                    was reached, │
        ▼                                            skips        │
       END (of subgraph, then → compliance)          summarize)  │
                                                                   │
       ◀───────────────────────────────────────────────────────┘
```

- **agent**: calls the LLM, offered its tool(s), free to call them or
  just ask a clarifying question.
- **tools**: executes whatever the LLM asked for, through the shared
  harness (retry once, then either loop back or escalate -- AND, since
  Milestone 9, also escalates if this specialist's already taken too many
  tool-calling rounds this request, even when every call succeeded). As
  of Milestone 12, this node is ASYNC -- tools come from the Argus MCP
  server (mcp_server.py/mcp_client.py) rather than direct Python imports,
  and MCP tools only support `.ainvoke()`. Failure detection is via
  `ToolMessage.status`, not a bare try/except -- MCP catches a tool's
  raised exception server-side and returns it as normal-looking content,
  so a naive try/except silently misses it (caught live, see Section 7).
- **summarize_decision**: ONE more LLM call, AFTER a real tool result
  exists, producing a typed decision for the audit trail. Deliberately
  skipped if no real tool was ever called (see the fabrication bug in
  Section 7) -- asking a clarifying question isn't a "decision."
- **human_escalation**: the automated system gives up and hands off to a
  person. As of Milestone 11, a REAL pause point -- calls `interrupt()`,
  which genuinely halts the whole graph (persisted via the checkpointer)
  until something resumes it with `Command(resume=<decision>)`. No longer
  a dead end.

---

## 5. File map

```
(repo root)
  Dockerfile           Milestone 15 -- containerizes the FastAPI app AND
                       the MCP server together (MCP uses stdio transport,
                       so the server is a subprocess, not a separate
                       network service -- one container, not several).
                       Runs as a non-root user, has a HEALTHCHECK against
                       /health. Verified building successfully on real
                       Docker via GitHub Actions (no Docker available in
                       the local dev sandbox).
  .dockerignore        Excludes .env from the image for real security
                       reasons -- Docker layers are immutable, a
                       "deleted" secret in a later layer is still
                       recoverable from an earlier one.
  .github/workflows/
    ci.yml               Fast tier -- every push/PR: pytest + a real
                         docker build + a real smoke test (boot the
                         container, poll /health, one real end-to-end
                         request). Asserts the response is WELL-FORMED,
                         not a specific business outcome (fixed after a
                         real run showed why -- Section 7).
    eval-gate.yml         Slow tier -- eval_gate.py (Milestone 14), on
                         manual dispatch + nightly only. NOT on every
                         push -- removed after a real run showed it
                         firing concurrently with ci.yml on the same
                         push caused real quota contention.

src/argus/
  state.py          The State schema every graph shares: messages, intent,
                     needs_escalation, decisions, audit_log.
  schemas.py         Pydantic models for structured LLM output:
                     IntentClassification (orchestrator's routing decision),
                     AgentDecision (a specialist's logged decision).
  llm.py             ONE function, get_llm(), building the Gemini client.
                     Every file that needs an LLM imports this instead of
                     constructing its own client.
  harness.py         Shared tool-execution infrastructure, used by every
                     specialist: retry-once-then-escalate, PLUS a
                     request-wide turn budget (Milestone 9) -- never used
                     directly, imported into each agent file. ASYNC as of
                     Milestone 12 (MCP tools only support .ainvoke()) --
                     detects failure via ToolMessage.status, NOT just
                     try/except (MCP swallows tool-side exceptions
                     internally; a bare try/except misses them entirely).
  mcp_server.py       The Argus MCP server (Milestone 12) -- wraps
                     tools/*.py's functions behind @mcp.tool(), launched
                     as a stdio subprocess. Mock logic unchanged; only how
                     specialists REACH it changed.
  mcp_client.py       Fetches tool objects from mcp_server.py at each
                     specialist's import time (asyncio.run() bootstrap,
                     same "build once" pattern as llm.py's client).
                     Explicitly passes GOOGLE_API_KEY to the subprocess
                     via StdioServerParameters' env field (Milestone 15
                     fix) -- the MCP SDK does NOT inherit the parent
                     process's environment by default, on purpose (a
                     security allowlist, see Section 7).
  guardrails.py      PII/KYC redaction (regex-based), runs as the very
                     first node in graph.py, before any LLM sees a raw
                     user message.
  skills.py           Reusable, versioned prompt fragments (Reason-Code,
                     Fraud Narrative, Coverage Lookup, Escalation) --
                     imported into specialist system prompts instead of
                     each one hand-writing similar phrasing separately.
  skill_evals.py       Each skill's eval set: rubric-check functions +
                     fixtures (pytest-covered, no LLM) + a live runner
                     (python -m argus.skill_evals) scoring real model
                     output from the real specialists.
  compliance.py       Structured decision summarization + the Send-based
                     fan-out to the audit log. Used by every specialist
                     (build_summarize_node) AND by graph.py (the fan-out
                     itself, which lives in the parent graph).
  rag.py             Hand-built retrieval: embed documents, embed a query,
                     rank by cosine similarity. Used only by Policy.
  graph.py           THE ORCHESTRATOR. classify_intent + routing to the
                     four specialists + the compliance fan-out. Compiled
                     WITH a checkpointer (MemorySaver, Milestone 11) --
                     every .invoke() now needs a thread_id in its config.
                     This is the file you'd run to talk to "the whole
                     system" directly in Python (its multi-turn memory
                     demo and real human-in-the-loop demo).
  api.py              FastAPI wrapper (Milestone 13) -- the real HTTP
                     boundary. POST /chat and POST /resume, both funneled
                     through _build_response() (reuses extract_text() from
                     Milestone 10). Builds the graph at TRUE top-level
                     module scope on purpose -- verified this is what
                     makes the MCP tool bootstrap (asyncio.run() at each
                     specialist's import time) survive being imported by
                     an async framework at all. Run with:
                     `uvicorn argus.api:app`.
  rag_evals.py         Milestone 14: hand-built RAGAS-style metrics for
                     Policy's retrieval -- context_precision (cheap, real
                     embeddings, pytest-covered) and judge_answer
                     (faithfulness + relevance via a separate LLM-as-
                     judge call, live-only).
  eval_gate.py          Milestone 14: the CI-ready gate --
                     `python -m argus.eval_gate` runs Milestone 10's
                     skill live-evals + this milestone's RAG evals,
                     returns exit code 0/1. Deliberately separate from
                     pytest -- pytest gates on cheap logic every push,
                     this gates on expensive QUALITY regressions, meant
                     for merge-time/nightly, not every commit.

  agents/               Each specialist fetches ITS tools via
                       mcp_client.get_mcp_tools(names=[...]) now
                       (Milestone 12), not direct imports from tools/*.py.
    fraud_investigation.py   Fraud specialist. Tool: get_fraud_score.
    underwriting_risk.py      Underwriting specialist. Tool: get_risk_grade.
    claims_triage.py          Claims specialist. Tools: get_claim_severity
                              AND flag_for_fraud_review (the handoff signal
                              -- intercepted in code, never really "executed").
    policy_customer.py        Policy specialist. Tool: search_policy_docs
                              (the only RAG-grounded specialist).

  tools/                Still directly imported by mcp_server.py (which
                       wraps them) AND by their own test files -- direct
                       import isn't gone, just no longer how SPECIALISTS
                       reach these functions.
    fraud_tools.py     get_fraud_score -- weighted rule (amount, prior
                       claims, odd hour) -> fraud_score/risk_band/drivers.
    risk_tools.py      get_risk_grade -- bucket rule on loan-to-income
                       ratio -> grade A-D.
    claims_tools.py    get_claim_severity -- lookup table by claim type.
                       Also flag_for_fraud_review (see claims_triage.py).
    policy_corpus.py   The synthetic FAQ documents Policy searches over.
    policy_tools.py    search_policy_docs -- wraps rag.py's VectorStore
                       around policy_corpus.py's documents.

  patterns/            Standalone, reusable pattern demos (Milestone 7) --
                       NOT part of the main request flow, built to prove
                       mechanics before using them for real elsewhere.
    fanout.py           Send-based parallel dispatch demo.
    reflection.py       generate -> critique -> revise loop demo.
    recursion_limit.py  Deliberately-infinite graph, proving the backstop.

tests/                  pytest suite -- deliberately only covers PURE LOGIC
                       (tools' math, harness retry logic, routers, cosine
                       similarity). Nothing that requires a live LLM call
                       is in here, to conserve free-tier API quota -- those
                       are exercised as manual `python -m ...` runs instead.
```

---

## 6. Why tools are mocks right now (and why that's deliberate, not lazy)

Every tool (`get_fraud_score`, `get_risk_grade`, `get_claim_severity`)
is a simple, deterministic Python function -- no trained model underneath.
This is on purpose: the whole point of the early milestones was to prove
the *agent/orchestration* layer works correctly before spending any effort
training real models. Because every tool has a fixed, typed contract
(inputs in, a specific shape of result out), swapping the mock for a real
trained model later touches ONLY that one tool file -- zero changes to any
agent, any prompt, any graph wiring. That swap is planned as a later
project phase, after the agentic build is further along.

---

## 7. Real bugs this project has actually hit and fixed (worth remembering)

- **Concurrent writes without a reducer crash the graph** (`InvalidUpdateError`)
  -- proven directly in Milestone 0, and again in Milestone 7's fan-out demo.
- **An escalation path that loops back to the agent can infinite-loop** on
  any deterministic failure (bad input data will just fail identically
  forever) -- caught by tracing the graph by hand BEFORE running it,
  Milestone 4. Fixed: escalation is a dead-end handoff, never a retry loop.
- **The compliance layer fabricated a decision** for a turn where no tool
  was actually called (Fraud asking a clarifying question got logged as
  `high_risk_flagged, confidence=0.95`) -- the exact "never fabricate"
  failure the harness exists to prevent, one layer higher. Fixed in
  Milestone 8 by checking tool NAMES against an explicit allowlist, not
  just "does any tool message exist in history."
- **Gemini rejects a request that ends in a model-turn message** ("Requests
  ending with a model turn are not supported") -- hit building the
  compliance summarizer, which handed it a message list ending in the
  specialist's own final answer. Fixed by appending an explicit trailing
  human turn.
- **Free-tier API quotas are per-model, not per-account** -- burned through
  a 20-requests/day cap on one Gemini model mid-Milestone-3, switched to a
  separate model with its own quota bucket.
- **`with_structured_output` silently drops token usage** unless you pass
  `include_raw=True` -- discovered while building token tracking in
  Milestone 9. Not a crash, just silently missing data, which is arguably
  worse (nothing tells you it's gone).
- **A reducer field that's fine stateless can be wrong once persisted.**
  `agent_turns` (an `operator.add` reducer, Milestone 9) kept accumulating
  ACROSS separate persisted conversation turns once checkpointing arrived
  (Milestone 11) -- turn 1 ended at 1, turn 2 ended at 3, never reset.
  Fixed by making it a plain field, explicitly reset every turn, instead
  of relying on the reducer. `decisions`/`audit_log`'s reducers stayed
  correct throughout -- a growing audit trail across a whole conversation
  IS what you want. Same reducer pattern, opposite correct scope --
  checkpointing is what forces you to actually notice the difference.
- **Adding `interrupt()` to a node breaks calling it as a bare function.**
  `human_escalation` could no longer be unit-tested by calling it directly
  once it started using `interrupt()` -- it needs LangGraph's runtime
  context. Fixed by testing it through an actual minimal graph instead.
- **A subprocess doesn't inherit the parent's PYTHONPATH.** `mcp_server.py`,
  launched as a stdio subprocess by the MCP client, couldn't find its own
  `argus` package (`ModuleNotFoundError`) until it was made to fix up its
  own `sys.path` at the top of the file -- fixed to not depend on however
  it happens to be launched, since that will look completely different
  under Docker/CI later.
- **MCP swallows tool-side exceptions -- a bare try/except misses them
  entirely.** The biggest bug of Milestone 12: FastMCP catches a tool's
  raised exception SERVER-SIDE and returns a normal (non-exceptional)
  result whose text happens to describe the error; a real human-in-the-
  loop test that used to pause correctly silently stopped pausing once
  MCP was wired in, because `needs_escalation` never got set. Fixed by
  detecting failure via `ToolMessage.status` (only available when
  invoking with the FULL tool-call dict, not just bare args) -- AND
  keeping the try/except too, since a PLAIN (non-MCP) tool behaves the
  OPPOSITE way and genuinely raises. Both mechanisms needed together;
  neither alone covers both tool types.
- **A paused nested subgraph's own earlier progress is invisible to the
  parent.** Milestone 13: a real HTTP escalation test showed the paused
  response's `agent_turns` as `0` while the interrupt payload (built
  from inside the paused node) showed `1`. Reproduced with a clean
  minimal example to confirm: the parent's `.ainvoke()` return reflects
  only what completed BEFORE the paused node started, not any progress
  the child subgraph made on its way there. Not a bug -- confirms
  `interrupt()` payloads have to be self-contained, not somewhere to
  economize by pointing at "just read it from state later."
- **A lesson learned once still has to be re-applied at each new call
  site.** Milestone 14's first version of `eval_gate.py` deferred
  specialist imports into an async function again -- the EXACT
  `asyncio.run()`-nesting trap `api.py` already fixed in Milestone 13,
  in a brand new file that hadn't existed yet when the lesson was first
  learned. Recognized the error immediately, fixed the same way (import
  at true top-level). The rule doesn't propagate on its own; every new
  file that imports these specialists has to apply it deliberately.
- **The MCP SDK does NOT inherit the parent process's environment --
  deliberately, a security allowlist.** Milestone 15: a Docker-simulation
  test (no `.env` file anywhere, only injected env vars) crashed with
  `GOOGLE_API_KEY not set` INSIDE the MCP server subprocess. Root cause,
  found by reading the actual SDK source: `get_default_environment()`
  only passes `HOME`/`LOGNAME`/`PATH`/`SHELL`/`TERM`/`USER` through by
  default -- an MCP server may be a third-party process, so the SDK
  refuses to forward secrets to it automatically. This RETROACTIVELY
  explains Milestone 12's `PYTHONPATH` fix too -- same root cause, never
  connected until this milestone, masked every earlier time because a
  real `.env` file was always on disk for the subprocess's OWN
  `load_dotenv()` to find independently. Fixed by passing
  `GOOGLE_API_KEY` explicitly via `StdioServerParameters`' `env` field
  (merges ON TOP of the safe defaults) -- deliberately not passing
  `os.environ` wholesale, which would defeat the allowlist's purpose.
- **A CI smoke test asserted a business outcome, not deployment health.**
  Milestone 16: a real CI run hit a transient tool failure; the harness
  correctly escalated instead of crashing (exactly the Milestone 4
  design, working as intended) -- and the smoke test FAILED anyway,
  because it grepped for `"paused":false` instead of checking the
  response was well-formed. A smoke test should assert the deployment is
  up and responding correctly, never a specific outcome that depends on
  a third-party API's availability at that exact second.
- **Two workflows on the same trigger compete for the same quota.**
  `eval-gate.yml` firing on every push to `main`, same moment as
  `ci.yml`, meant both hit the same free-tier API key concurrently --
  the direct cause of the failure above. Removed the push trigger;
  manual + nightly only, which was the actual intent all along.
- **An unauthenticated polling loop against a REST API burns its rate
  limit fast.** Checking GitHub Actions run status via repeated
  `curl`+`sleep` calls exhausted the 60-req/hour unauthenticated limit in
  one loop -- should have used one longer-interval check, or just pointed
  at the Actions UI directly rather than polling from here at all.

---

## 8. Status as of last update

**Milestone 15+16 (Dockerize + CI/CD) complete, INCLUDING real local
verification** -- pulled forward and merged on purpose, since verifying
the Dockerfile genuinely required the CI pipeline first (no Docker
locally in this sandbox). Pushed the project to a real GitHub repo
(`waqar9425/bfsi-genai-project`), GitHub Actions confirmed the Dockerfile
builds/runs on real infrastructure, then Docker Engine got installed for
real on the actual dev machine (WSL2, no `systemd` -- daemon started
directly via `sudo dockerd`, full story in `CONTAINERS_AND_CICD_CONCEPTS.md`)
and `argus:local` was built and run for real, hit with a real `curl`,
answered a real grounded question, correctly. Every piece that made that
work -- async MCP bootstrap, the env-var fix, `--host 0.0.0.0`, the
harness, compliance logging -- held together on the first real end-to-end
run. Real bugs found from actual CI failures and fixed (both above); this
closes out the planned agentic-build roadmap (Milestones 0-16) -- next is
Phase D, swapping the mocked tools for real trained models.

**Everything through Milestone 14 stays complete and unaffected:** All four specialists real and working, shared
harness (retry/escalation + a correctly PER-TURN turn budget, now async
and MCP-aware), one cross-agent handoff (Claims → Fraud via `Command`),
real RAG (Policy), structured per-agent decisions fanned out to an audit
log (Compliance), PII/KYC redaction at the entry point, thread-wide token
usage tracked across every LLM call, shared/versioned prompt fragments
(Skills) with their own rubric-based eval sets, the whole graph
checkpointed with real multi-turn memory and a genuine human-in-the-loop
pause point, every tool reached through a real MCP server/client, a real
FastAPI HTTP layer (`POST /chat`, `POST /resume`), and now (Milestone 14)
a CI-ready eval gate (`python -m argus.eval_gate`) combining skill
live-evals with hand-built RAGAS-style RAG evals (context precision +
LLM-as-judge faithfulness/relevance) into one command with a real
pass/fail exit code -- verified both directions, that it passes when
things are actually fine AND fails when they're not.
54 pytest tests passing (pure logic + checkpointing/interrupt mechanics +
API response-shaping + the eval gate's own pass/fail logic + RAG
retrieval hit-rate, still zero LLM-generation calls for the bulk of the
suite -- only `test_rag_evals.py` makes real embedding calls).

**Not built yet:** Langfuse observability. That's it from the original
agentic-build roadmap -- next up is Phase D, swapping the mocked tools
in `tools/*.py` for real trained models, per the blueprint's Section 07.
