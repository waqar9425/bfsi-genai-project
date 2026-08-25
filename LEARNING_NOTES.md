# Argus — Learning Notes (interview crib sheet)

Running notes as we build. Each entry: the concept, why it's tested, and where it lives in this repo.

---

## Milestone 0 — LangGraph mechanics (no LLM)

**Files:** `src/argus/state.py`, `src/argus/graph.py`

### Concept: State + reducers
- `State` is a `TypedDict` (could also be a Pydantic model — tradeoffs below).
- Every node returns a **partial update**, not the full state. LangGraph merges it in.
- Default merge = last-write-wins (overwrite). Fields that should *accumulate*
  (like conversation history) need an explicit reducer via `Annotated[type, reducer_fn]`.
- `add_messages` is LangGraph's built-in reducer for chat history: it appends,
  and also handles de-duping by message ID for updates.

**Interview question:** *"If two parallel nodes both write to the same state
key, what happens?"*
→ Depends on whether it's the *same step* and whether the key has a reducer.
  - **Sequential** writes to a non-reducer key (node A runs, later node C
    runs and also sets it): silent last-write-wins. Normal, expected.
  - **Concurrent** writes to a non-reducer key (two nodes in the *same*
    parallel step both return the same key): LangGraph raises
    `InvalidUpdateError: Can receive only one value per step` — it refuses
    to guess, it crashes. Verified empirically, see git history / this repo.
  - **Reducer** fields (`add_messages`, `operator.add`, etc.): concurrent
    writes are fine — reducer combines all contributions deterministically.

This is *the* reason to think carefully about reducers before adding
parallel branches — a fan-out graph that "worked in testing" can crash the
first time two branches happen to touch the same plain field. Know this
cold, it's a favorite "have you actually built one of these" filter
question.

**Interview question:** *"TypedDict vs Pydantic BaseModel for graph state — which and why?"*
→ TypedDict: zero runtime validation cost, matches what most LangGraph
examples use, fine when nodes are trusted internal code. Pydantic: you get
validation/coercion at graph boundaries (e.g. if state can be populated from
an external API request), better for catching bugs early, slightly more
overhead. Argus uses TypedDict for internal graph state but Pydantic for
node *outputs* that need structured-output validation from the LLM (see
Milestone 1) and for the MCP tool contracts (typed request/response, must
validate).

### Concept: Nodes are just functions
- Signature: `(state: State) -> dict`. No base class, no decorator required
  (though `@tool` exists for tool functions specifically — different thing).
- Side effects (printing, calling an API, calling an LLM) are fine inside a
  node — the node is the unit of work.

### Concept: Conditional edges = routing
- `add_conditional_edges(source_node, router_fn, mapping)`.
- `router_fn(state) -> str` returns a key into `mapping`, which resolves to
  the actual next node name.
- This is the entire mechanism behind the Orchestrator's routing to Claims /
  Fraud / Underwriting / Policy specialists. No hidden magic — just a
  function returning a string.

**Interview question:** *"How would you unit-test a conditional edge?"*
→ Router functions are pure functions of state → string. Test them directly
without ever invoking the graph: `assert route_by_intent({"intent": "fraud", ...}) == "fraud"`.
This is a real advantage of the explicit-graph design over a hidden agent
loop — the routing logic is a plain testable function.

**Vocabulary trap:** a *router function* (like `route_by_intent`) is NOT a
*tool*. A router is plain deterministic Python wired into a conditional
edge — the LLM never sees it or chooses to call it. A *tool* (Milestone 4+)
is a function exposed to the LLM via function-calling, which the model
itself decides to invoke (fraud-scoring model, retrieval, Snowflake lookup
— all MCP-wrapped in Argus). Conflating the two is a common early mistake.

---

## Milestone 1 — real LLM orchestrator, structured output

**Files:** `src/argus/schemas.py`, `src/argus/llm.py`, `src/argus/graph.py`

### Concept: structured output IS tool-calling
`with_structured_output(Schema)` is not "ask nicely for JSON." Two real
mechanisms, provider-dependent:
1. **Forced tool-calling** (what Gemini/most providers do via LangChain):
   your Pydantic schema → JSON Schema → registered as a fake "tool" → model
   is *forced* to call it (`tool_choice="required"` equivalent) → the tool
   call's arguments ARE your parsed object. Google's own logs call this
   "automatic function calling" (AFC) — same mechanism, their name for it.
2. **True constrained decoding / JSON mode**: sampling itself is restricted
   so invalid-schema output is structurally impossible (e.g. OpenAI strict
   `response_format=json_schema`). Stronger guarantee, less universal.

**Interview question:** *"What's the difference between structured output
and tool calling?"*
→ Trick question in a good way — for most providers there isn't one at the
wire-protocol level. Structured output is tool-calling with exactly one
tool and a forced call. When Argus's Fraud Agent later calls the real
fraud-scoring model as a tool, and when the Orchestrator classifies intent,
it's the same underlying request shape — one *returns* data to the caller,
the other *requests* an external action, but the mechanism is identical.

**Interview question:** *"What happens when the model's output fails
schema validation?"*
→ LangChain raises a validation error, it's not silently swallowed or
auto-corrected. This is exactly why a harness needs an explicit retry
layer around structured-output calls (see blueprint's Harness section) —
without it, one malformed response crashes the whole request instead of
degrading gracefully.

### Design signal from testing, not just a coding lesson
Fed the classifier an intentionally ambiguous message (accident report +
fraud suspicion in one sentence). It picked one intent and explained why —
but a flat single-shot router can only ever hold one intent per turn. A
specialist that detects a second intent mid-conversation (Claims Triage
spotting fraud cues) needs to *hand off*, which is a different LangGraph
pattern (`Command`-based handoffs / graph-of-graphs) than top-level routing.
Flagged for when Claims Triage and Fraud Investigation both exist for real.

---

## Milestone 2 — real (optional) tool calling: the ReAct loop

**Files:** `src/argus/tools/fraud_tools.py`, `src/argus/agents/fraud_investigation.py`

### Concept: bind_tools (optional) vs with_structured_output (forced)
- `with_structured_output(Schema)`: ALWAYS calls, exactly one fake tool,
  every invocation. Used for routing (Milestone 1) -- we always need an
  intent.
- `bind_tools([real_tool, ...])`: model is OFFERED tools, free to call zero,
  one, or several, or just reply in text. Used for specialist agents that
  need to gather info before acting.
- The `@tool` decorator turns a plain function into a schema the model
  sees: name + docstring (as description) + args derived from type hints.
  The docstring/type hints ARE the interface -- a vague docstring is a
  vague API to the model, same as it would be to a human caller.

### Concept: the ReAct loop, built from two prebuilt LangGraph pieces
`agent` node (calls the model with tools bound) --conditional edge--> if
`response.tool_calls` non-empty: `tools` node (executes them, appends
`ToolMessage`s) --> back to `agent` --> repeat until no more tool calls -->
`END`.
- `ToolNode(tools)`: prebuilt node, executes whatever's in the last
  message's `.tool_calls`.
- `tools_condition`: prebuilt conditional-edge function, checks
  `len(last_message.tool_calls) > 0`, returns `"tools"` or `END` directly
  (no mapping dict needed, unlike Milestone 1's router -- these are real
  node names/sentinels already, not our own made-up strings).

**Verified empirically, not just asserted:**
- The model correctly reasoned "4th claim" -> `prior_claims_count=3`
  (argument *construction*, not blind extraction) before calling the tool.
- Given an under-specified prompt, the model called ZERO tools and asked
  clarifying questions instead -- nobody wrote an `if` statement for this;
  `tools_condition` simply didn't route to `"tools"` because
  `response.tool_calls` was empty. This is the model itself deciding.
- The dict returned by the tool (`response.model_dump()`) got
  auto-serialized to a JSON string as the `ToolMessage` content -- this is
  literally the boundary where a typed Python object becomes text the
  model reads back.

**Gotcha, real not hypothetical:** Gemini's `message.content` came back as
`list[dict]` (content blocks + metadata), not a plain `str`. Any code
assuming `.content` is always a string breaks the moment a provider does
this. Don't hardcode that assumption anywhere in the harness later.

**Interview question:** *"Why does the system prompt get re-prepended in
`call_model` on every loop iteration instead of being stored in `state["messages"]` once?"*
→ State holds conversation *data* (what's been said); the system prompt is
agent *configuration* (how the agent should behave), constant across the
whole run. Storing it in state would mean it flows through the checkpointer,
gets persisted, and re-sent every time state is reloaded -- mixing
configuration with data. Keep it as a constant applied at call-time instead.

### Deliberate scope cut, noted for later
The tool is a plain LangChain `@tool`, not exposed over real MCP transport
yet. Matches the blueprint's own "mocks/simple contracts before
infrastructure" order -- MCP wrapping is a transport-layer concern we'll
add once the core agent loop is proven, not a prerequisite for learning
tool-calling itself.

---

## Milestone 3 — subgraphs: a compiled graph IS a node

**Files:** `src/argus/graph.py` (fraud stub replaced with `build_fraud_agent()`)

### Concept: compiled graph == Runnable == valid node
`StateGraph(...).compile()` returns something that implements the same
`.invoke()` contract as any node function. `add_node("fraud", _fraud_agent)`
works with ZERO wrapper code -- no special "subgraph" API to learn.

Two real patterns, know which one you're in:
- **Shared schema** (us, right now): parent and child both import the same
  `State`. Drop the compiled child straight in as a node. Parent state
  flows in unmodified; child's updates merge back via the same reducers,
  across the parent/child boundary, transparently.
- **Divergent schema** (Argus later, e.g. Claims Triage growing a private
  `extracted_claim_details` field): needs an explicit wrapper node that
  maps parent state -> child state going in, child state -> parent update
  coming out. Not built yet -- flagged for whenever a specialist's
  internal state actually diverges from what the orchestrator needs to see.

**Interview question:** *"How does LangGraph handle a subgraph's internal
node names (e.g. `"agent"`, `"tools"`) not colliding with the parent
graph's node names?"*
→ Each subgraph invocation runs in its own internal namespace for
checkpointing/step-tracking; the parent graph only ever sees the single
node name it was registered under (`"fraud"`). No manual namespacing needed.

### Verified: non-determinism is real, not a testing artifact
Same "4th claim" phrasing produced `prior_claims_count: 3` in one run and
`4` in another (Milestone 2 vs. Milestone 3 runs, same prompt, low
temperature). Confirms: never trust an LLM's arithmetic/extraction as the
source of truth for anything that needs to be deterministic -- that's
exactly why the actual scoring logic lives in a plain, unit-tested Python
function the model can only *call*, never *reimplement*.

### Operational lesson: free-tier quotas are model-specific and can be tiny
Hit `429 RESOURCE_EXHAUSTED` on `gemini-flash-latest` (resolves to
`gemini-3.7-flash`) mid-milestone -- free tier allows only **20
requests/day** for that model, tighter than expected for a "generous free
tier" (newer/preview-ish models get the tightest caps). Switched to
`gemini-flash-lite-latest`, a separate quota bucket entirely, confirmed
working immediately. Lesson: rate limits in this space are per-model, not
per-account -- if one model's quota is exhausted, a different model on the
same key/account may still have headroom.

---

## Milestone 4 — the harness: retry, then escalate, never fabricate

**Files:** `src/argus/harness.py`, `src/argus/tools/risk_tools.py`,
`src/argus/agents/underwriting_risk.py`, `fraud_investigation.py` (retrofitted)

### Concept: harness = agent-independent, built once, reused by every specialist
`build_tools_node(tools)` + `route_after_tools` + `human_escalation` +
`tools_present` replace the bare `ToolNode`/`tools_condition` pair in BOTH
Fraud and Underwriting -- same four pieces, two different agents. This is
the literal code form of the blueprint's Section 06 claim: the harness
wraps every tool call, independent of any one agent's own logic.

### Concept: retry helps transient failures, not deterministic bugs
`_invoke_with_retry` retries once, then gives up -- it can't tell a
network blip from a real bug, so it just enforces a fixed budget (2
attempts) and lets the *caller* (the router) decide what happens next. A
deterministic failure (bad input data, e.g. `annual_income=0`) will fail
the exact same way on the retry -- retrying doesn't fix logic errors, it
just wastes one extra call before failing anyway.

### Design bug caught and fixed BEFORE running: escalation must not loop
First draft routed `human_escalation -> "agent"` (loop back and try again).
Traced through by hand: a deterministic failure (bad income data) would
retry, fail, escalate, loop back, retry with the SAME bad data, fail
again, escalate again -- forever. Fixed to `human_escalation -> END`:
escalation is a HANDOFF, not a retry. The automated graph stops; a human
takes over from there. Caught by tracing the graph by hand before ever
running it, not by hitting an infinite loop live -- the cheaper way to
catch this class of bug.

**Interview question:** *"How do you prevent a retry/escalation graph from
looping forever?"*
→ Two independent guards, not one: (1) a hard retry budget inside the
retry wrapper itself (`MAX_ATTEMPTS`, not "retry until it works"), and (2)
an escalation edge that terminates the automated path rather than looping
back into the same failure-prone node. Relying on just the retry budget
isn't enough if the graph topology loops the *whole agent* back around
after a failure -- the budget resets every time you re-enter the node.

### Verified: the harness is testable with ZERO LLM calls
All 5 new harness tests (retry-succeeds, retry-recovers, retry-exhausted,
both router branches) run against a hand-built flaky fake tool -- no
model, no graph traversal, no API quota spent. Same principle as testing
`route_by_intent` in Milestone 0, now proven on something stateful
(a call counter) instead of a pure function. This is *why* it's worth
architecting retry/escalation as plain functions operating on
`tool_calls` lists rather than burying the logic inside a bigger
LLM-dependent node -- the expensive, slow, flaky part (the model) and the
cheap, fast, deterministic part (the harness) stay separately testable.

---

## Addendum — deeper context on Milestones 1-3 (added retroactively)

**Three separate retry layers exist in this stack, don't conflate them:**
1. HTTP-client-level (Google SDK's internal `tenacity` retry on raw
   connection issues) -- invisible, not ours, not configurable by us.
2. LangGraph node-level (`RetryPolicy`, opt-in per node) -- we never
   configured this; without it, a node exception propagates straight out
   and crashes the whole `.invoke()` call (observed directly in the
   Milestone-1 and Milestone-3 quota crashes).
3. Our own harness retry (Milestone 4) -- tool-call-level, business-logic
   aware, drives a typed outcome (`needs_escalation`) rather than just
   re-attempting.

**`tool_choice` has four real positions**, we've now used two without
naming the set: `"auto"` (model decides freely -- our specialists),
`"required"`/`"any"` (must call something, model picks which),
`"none"` (disabled), forced-to-one-tool (what `with_structured_output`
does -- our orchestrator).

**ReAct = Reasoning + Acting** (Yao et al. 2022). Originally interleaved
free-text "Thought/Action" reasoning with actions, parsed via fragile
regex, because models had no native function-calling. Modern tool-calling
APIs are ReAct made a first-class model capability -- our
`agent -> tools -> agent` loop IS ReAct, running on structured calls
instead of parsed text.

**LangGraph's engine is Pregel (BSP: Bulk Synchronous Parallel)** --
execution proceeds in super-steps; all ready nodes run, their updates
merge via reducers, then the next super-step's ready set is computed. This
is WHY concurrent writes to a non-reducer key hard-crash (Milestone 0) --
BSP requires a well-defined merge per step, "last one wins" isn't
well-defined when execution order within a step isn't guaranteed.

**Compiled graphs work as nodes via structural typing, not a special
LangGraph feature** -- `.compile()` returns something satisfying the same
minimal Runnable interface (`invoke`/`stream`) any node needs. No
"subgraph" special case exists in the scheduler; it just calls `.invoke()`
on whatever's registered.

**Known gap, not yet built: checkpointing.** Every test so far starts
from a blank state -- nothing persists across separate `.invoke()` calls.
A `checkpointer` + `thread_id` is what enables real multi-turn memory AND
genuine human-in-the-loop (pausing at `human_escalation` for a real
person to act, instead of it being a dead-end message). Our escalation
design has been quietly assuming a human "takes over outside the graph" --
checkpointing is what would make that literal.

---

## Milestone 5 — Command handoffs: solving the multi-intent problem for real

**Files:** `src/argus/tools/claims_tools.py`, `src/argus/agents/claims_triage.py`, `graph.py`

### Concept: Command lets a NODE decide routing dynamically, not a pre-declared edge
Every router so far (`route_by_intent`, `tools_present`, `route_after_tools`)
is a separate function, declared ahead of time, wired in via
`add_conditional_edges`. `Command(update={...}, goto="node_name")` is a
different shape entirely: the NODE ITSELF, in the middle of doing its own
work, decides both a state update AND the next hop, in one return value --
no separate router function needed, and critically, the decision can
depend on things only knowable mid-execution (here: what the model
actually said this turn), not just on state set by a previous node.

### Concept: Command(graph=Command.PARENT) crosses the subgraph boundary
A subgraph's own conditional edges can only route among nodes IT knows
about. `Command(goto="fraud", graph=Command.PARENT)` is the specific
mechanism for a node inside a subgraph (Claims Triage) to jump to a node
in the PARENT graph (the orchestrator) -- verified empirically with an
isolated child/parent example before writing any real code, then proven
for real in the live run: the handoff bypassed Claims' own
`agent -> tools -> END` shape entirely and landed straight in Fraud's
agent node, one level up.

**Real cost of this pattern, not just a benefit:** Claims Triage is no
longer fully self-contained. It now has an implicit dependency on a node
literally named `"fraud"` existing one level up in whatever parent graph
it's wired into -- it can't be meaningfully tested standalone for the
handoff path the way Fraud and Underwriting could be. Coupling, traded for
capability -- name this tradeoff explicitly if asked "any downsides to
Command-based handoffs?" in an interview.

### Concept: intercept a "decision" tool call BEFORE normal tool execution
`flag_for_fraud_review` is offered to the model via `bind_tools`, but
deliberately excluded from `build_tools_node`'s tool list. `call_model`
inspects `response.tool_calls` for that specific name and short-circuits
with a `Command` before the harness's normal tools node ever sees it. Two
different tools, offered in the same `bind_tools` call, handled by two
completely different code paths based on which one the model picked.

**Real gotcha, proven not hypothetical:** every tool_call in message
history needs an eventual matching `ToolMessage`, even one you're
intercepting rather than "really" executing -- built the `ack` message by
hand for exactly this reason, and the live run confirms why it mattered:
Fraud's very next LLM call needed clean, fully-resolved message history to
succeed. Skipping the ack would have left a dangling unanswered tool call
in the history handed to a completely different agent.

**Known, documented simplification:** if the model called BOTH
`get_claim_severity` AND `flag_for_fraud_review` in the same turn, only
the handoff is handled -- the severity call would be silently dropped
(never answered, never executed). Same shape of edge case as the
Milestone-2 "two claims in one sentence" drill question, still unhandled,
now for real in shipped code. Worth fixing if this were headed to
production; left as a named gap here, not silently ignored.

---

## Milestone 6 — real RAG: retrieval as a fundamentally different tool

**Files:** `src/argus/rag.py`, `src/argus/tools/policy_corpus.py`,
`src/argus/tools/policy_tools.py`, `src/argus/agents/policy_customer.py`

### Concept: retrieval mechanics, built by hand deliberately
Embed every document once (batched: `embed_documents`, one call for the
whole corpus, not N calls); embed the query the same way at request time;
rank by **cosine similarity** -- the angle between vectors, not distance,
because embedding models are trained so meaning correlates with direction,
not magnitude (verified: `[1,0]` vs `[5,0]` scores 1.0 -- same direction,
5x the length, identical similarity). A real vector database (FAISS,
pgvector, Pinecone) does this exact comparison at scale via
approximate-nearest-neighbor indexing (HNSW, IVF) instead of brute force --
different SCALE, not different math. Built the brute-force version by
hand specifically so the mechanics aren't a black box.

### Concept: a retrieval tool is qualitatively different from a scoring tool
`get_fraud_score`/`get_risk_grade`/`get_claim_severity` return one
definitive answer the model relays. `search_policy_docs` returns raw,
ranked, sometimes-irrelevant excerpts the model must READ, judge, and
synthesize an answer from -- "grounding," not "computation." This is why
the system prompt has an explicit "if results don't answer the question,
say so, don't guess" instruction -- ungrounded hallucination on missing
retrieval is one of the most common real RAG failure modes.

### Verified live, not scripted: emergent retry-and-rephrase behavior
Asked about "meteor strike" coverage (not in the corpus). The model:
searched once, judged the results weren't actually relevant, **retried
with a rephrased query** ("act of God exclusion" -- a real insurance term
it reached for on its own, never in any prompt), judged those irrelevant
too, and only then said the documents don't cover this -- instead of
confidently inventing a plausible-sounding exclusion clause. Nobody coded
"retry with different phrasing if results look weak" -- this fell out of
the system prompt's instruction plus the model's own judgment of what it
retrieved.

**Real, measurable signal from that run:** the on-topic case's top
relevance score was 0.764; the off-topic case's best score across two
searches topped out around 0.65. A production system wouldn't rely purely
on the model eyeballing borderline scores -- a hard relevance threshold
(e.g. refuse to surface anything below ~0.7) is a standard additional
guardrail, not implemented here, but the score gap in this exact run is
what such a threshold would act on.

**Interview question:** *"How would you evaluate a RAG pipeline?"*
→ Two separate axes, don't conflate them: retrieval quality (did the
right documents come back? -- precision/recall against a labeled set) and
generation quality given good retrieval (did the model use them
correctly, cite them, avoid contradicting them? -- faithfulness). The
blueprint's own plan names RAGAS specifically for this
(faithfulness, answer relevance, context precision) -- not built here yet,
but know the framework name and what it measures.

---

## Milestone 7 — Graph & Loop Engineering Patterns

**Files:** `src/argus/patterns/fanout.py`, `recursion_limit.py`, `reflection.py`

### Part 1: Send -- dynamic parallel fan-out
A conditional-edge function can return `list[Send(node_name, state)]`
instead of a single node-name string -- each `Send` is an independent
parallel invocation of that node with its OWN local state (verified: each
`log_one` branch only ever saw the one decision it was given, not the
whole list). This is LangGraph's "map" primitive.

**The load-bearing fact, proven twice:** fan-out is safe ONLY because the
target field has a reducer. Removed the reducer from `audit_log` and got
the EXACT SAME `InvalidUpdateError` as Milestone 0's original
concurrent-write experiment, character for character. Send-based fan-out
isn't a new failure mode to learn -- it's Milestone 0's "concurrent writes
need a reducer" lesson, now triggered on purpose instead of by accident.
This is the pattern Milestone 8 uses for real: every specialist's decision
fans out to Compliance & Audit in parallel.

### Part 2: reflection loop -- a different loop SHAPE, not just another loop
Every loop since Milestone 2 is generate-ACT-observe (model acts on the
world via a tool, reacts to the result). Reflection is generate-CRITIQUE-
revise: the model acts on its OWN prior output. Built as
`generate -> critique -> (route: END if approved, else back to generate,
capped by MAX_REFLECTION_ATTEMPTS)`.

**Verified live, not just structurally:** forced a task likely to fail
first-pass critique ("we believe the claimant is lying..."). Critique
caught two real problems (vague language, implied bad-faith accusation
without evidence) and rejected. The SECOND draft, generated with that
specific feedback, fixed exactly those two things -- reframed around
"independent verification records" instead of implying dishonesty, and
was approved. Feedback measurably changed the output, not cosmetically.
Confirms the loop-back edge actually works, not just the happy-path
single-pass case (which is what the FIRST test run showed, by accident --
first draft passed immediately, so the revision path was unverified until
deliberately forced with a harder task).

**Interview framing:** this generalizes the harness's `MAX_ATTEMPTS`
(Milestone 4) from "retry one tool call" to "retry a whole generate/judge
cycle" -- same underlying principle (a loop needs an explicit, enforced
budget) applied one level up the stack.

### Part 3: recursion_limit -- the last-resort, graph-wide backstop
Built a graph with NO exit condition on purpose (`ping <-> pong` forever)
to watch `GraphRecursionError` actually fire, rather than take the safety
net on faith (it was mentioned as existing back in Milestone 2, never
triggered until now).

**The distinction that matters:** harness `MAX_ATTEMPTS` is OUR code,
enforcing a budget on ONE specific known failure mode (a flaky tool).
`recursion_limit` is LangGraph's own backstop, graph-wide, catching ANY
runaway loop regardless of cause -- a router bug, a model that keeps
calling tools forever, anything. Belt and suspenders: the harness budget
should never need recursion_limit's help in practice, but recursion_limit
is what protects you the day the harness budget itself has a bug.

---

## Milestone 8 — structured decisions + Compliance & Audit

**Files:** `src/argus/compliance.py`, `schemas.py` (`AgentDecision`), `state.py`
(`decisions`, `audit_log`), all four specialists (new `summarize` node), `graph.py`

### Shape
```
  each specialist: ... agent/tools loop ... --no tool_calls--> summarize_decision --> END (of subgraph)
                                                                       |
                                                          appends to state["decisions"]
                                                                       v
  parent graph:  fraud/claims/underwriting/policy --> compliance --Send fan-out--> log_decision (xN, parallel)
                                                                                          |
                                                                             appends to state["audit_log"]
```
`tools_present` (harness.py) is UNCHANGED -- still only ever returns
`"tools"` or `END`. What changed is the mapping passed to
`add_conditional_edges`, remapping `END -> "summarize"` per specialist.
Verified in Milestone 7 that a returned sentinel can be remapped this way
before relying on it here.

### The real bug this milestone caught, not just a feature it added
First live run: Claims handed off to Fraud (Milestone 5's path). Fraud
asked a clarifying question -- no tool call, no score computed. Compliance
still logged: `decision='high_risk_flagged' confidence=0.95`, citing
narrative details from the conversation, not anything a tool computed.

**This is the exact failure the harness (Milestone 4) exists to prevent
-- "never fabricate, escalate/ask instead" -- reappearing one layer
higher, in the audit trail meant to be the trustworthy record.** A
naive fix ("skip summarizing if there's no tool_calls in the last
message") isn't enough either -- that's already true for every
info-gathering turn AND every real final answer, doesn't distinguish them.

Real fix required tagging every `ToolMessage` with its actual tool name
(`name=call["name"]`, added to `harness.build_tools_node`) and checking
against an explicit per-specialist allowlist (`build_summarize_node(name,
real_tool_names)`) -- NOT just "does a ToolMessage exist anywhere in
history." That distinction mattered concretely: Claims' handoff ack IS a
ToolMessage (for `flag_for_fraud_review`), sitting in history before Fraud
ever runs -- a naive "any tool message" check would have been fooled by it
too. Verified the fix against the exact scenario that broke, not just the
general case: `decisions: []`, printed `"no tool was called yet --
skipping decision summary, nothing real to audit"`.

**Interview framing:** "never fabricate" isn't a one-time design decision
you make once in the harness and forget -- it's a property that has to be
re-verified at every layer that summarizes or reports on agent behavior.
An audit/compliance layer is especially dangerous to get this wrong in --
it's the layer whose entire job is being trustworthy.

### Second real bug, smaller but also live not theoretical
Gemini rejected `summarize_decision`'s first request outright: `"Requests
ending with a model turn are not supported."` -- the message list handed
to it ended with the specialist's own final AI answer, a model-turn
message, with nothing for the API to respond to. Fixed by appending an
explicit trailing human turn. Provider-specific constraint, not universal
across all chat APIs -- worth knowing this class of constraint exists,
not just this one instance of it.

---

## Milestone 9 — Guardrails: PII redaction + turn/token budgeting

**Files:** `src/argus/guardrails.py`, `state.py`, `harness.py`, `llm.py`
(`get_token_usage`), `compliance.py`, `graph.py`, all four specialists

### Part A: PII redaction, via a reducer nuance not used before
`add_messages` doesn't just append -- if the returned message's `.id`
matches an EXISTING message already in state, it REPLACES it instead
(upsert-by-id). Verified empirically before relying on it. This is how
`redact_pii_node` edits the user's raw message in place: read
`state["messages"][-1].id`, build a replacement `HumanMessage` with the
SAME id and redacted content, return it. Wired as the very first node
(`START -> redact_pii -> classify_intent -> ...`) so no LLM call, ever,
sees unredacted PII -- verified live: the final message transcript shows
`[REDACTED_SSN]`/`[REDACTED_EMAIL]`, not the raw values.

### Part B: turn budget -- a second, distinct reason to escalate
`route_after_tools` (harness.py) now checks TWO independent conditions,
same destination: `needs_escalation` (a tool failed after retries) OR
`agent_turns >= MAX_AGENT_TURNS` (too many tool-calling rounds, even if
every individual call succeeded). Scoped REQUEST-WIDE, not per-specialist
-- if Claims hands off to Fraud, both specialists' tool rounds add to the
same total (`agent_turns: Annotated[int, operator.add]`). Named as a
simpler, honestly-scoped choice over trying to reset the counter per
specialist across a Command handoff.

**Interview framing:** this is a SOFTER, business-aware sibling of
`recursion_limit` (Milestone 7) -- our own budget escalates gracefully to
a human; `recursion_limit` is LangGraph's own blunt, graph-wide backstop
that hard-crashes with `GraphRecursionError`. Belt and suspenders, same
relationship as harness `MAX_ATTEMPTS` vs `recursion_limit` from Milestone 7.

### Part C: token tracking -- and a real gotcha in with_structured_output
Verified empirically: `with_structured_output(Schema)` returns ONLY the
parsed object by default -- `usage_metadata` isn't accessible AT ALL from
it, silently. Fixed with `include_raw=True`, which changes the return
shape to `{"raw": AIMessage, "parsed": Schema, "parsing_error": ...}`.
Bonus: this is ALSO the mechanism that answers "what happens when
structured output fails validation" from Milestone 1 -- without
`include_raw`, a parse failure raises immediately; with it, you get the
error object handed to you instead of a crash, and can decide what to do
(not yet built out here, just now possible). Applied to both
`classify_intent` (graph.py) and `summarize_decision` (compliance.py) --
the two structured-output call sites. `bind_tools`-based specialists never
had this problem -- they already return the raw `AIMessage` directly.

Tracked via `total_tokens_used: Annotated[int, operator.add]`, reported by
every LLM-calling node. Deliberately NOT enforced as a hard cap yet
(tracking/observability first, matching "mocks before infrastructure") --
a natural extension once real cost matters more than free-tier quota does.

---

## Milestone 10 — Agent Skills: shared, versioned prompt fragments + eval sets

**Files:** `src/argus/skills.py`, `skill_evals.py`, all four specialists, `harness.py`

### Concept: a Skill is a reusable prompt fragment, not a new mechanism
No new LangGraph API here -- a `Skill` is a plain frozen dataclass
(`name`, `version`, `instructions`), and each specialist's system prompt
is now ASSEMBLED (base role text + skill fragments joined with `\n\n`)
instead of hand-written as one long string. `REASON_CODE_SKILL` is
imported into BOTH `fraud_investigation.py` and `underwriting_risk.py` --
proven with a real test, not just claimed: `test_skills.py` asserts the
literal `REASON_CODE_SKILL.instructions` string is a substring of both
specialists' actual system prompts.

### Concept: two-layer eval sets, same offline/live split as everything else
`skill_evals.py` has deterministic rubric-check functions (pure Python
string/regex checks) scored against hand-written good/bad fixtures --
zero LLM cost, pytest-covered (`test_skill_evals.py`). Necessary but not
sufficient: those tests only prove the RUBRIC correctly tells good from
bad on examples I wrote by hand. The real proof is the live runner
(`python -m argus.skill_evals`), which feeds genuine model output through
the actual specialists into the SAME rubric functions.

**Verified live, and the output structure itself changed, not just
tone:** Fraud's live output literally organized itself as "**SIU
Investigator Note**" with a "**Recommendation:**" line -- matching
`FRAUD_NARRATIVE_SKILL`'s instructions structurally, not just
superficially. All 3 live evals passed (fraud_narrative, reason_code,
coverage_lookup) against real model output.

**Interview framing:** this is what "prompt regression testing" actually
looks like in a production system -- not vibes-based "the output looks
fine," a rubric function with fixtures, the same discipline as any other
regression test, just checking text properties instead of return values.
This is also literally what a later CI eval-gate (Milestone 15 in the
roadmap) would run on every PR that touches a skill's instructions.

### A small but real fix along the way: extract_text()
Reused the Milestone-2 gotcha (Gemini's `message.content` is sometimes
`list[dict]`, not `str`) -- `skill_evals.py`'s `extract_text()` normalizes
either shape before a rubric check ever sees it, rather than every rubric
function having to special-case content shape itself.

---

## Milestone 11 — Checkpointing & memory: closing the gap named since Milestone 3

**Files:** `graph.py` (`checkpointer=MemorySaver()`), `harness.py`
(`interrupt()`), `state.py`/`guardrails.py` (the `agent_turns` fix),
`test_checkpointing.py`

### Concept: a checkpointer + thread_id turns "stateless calls" into "persisted conversations"
`g.compile(checkpointer=MemorySaver())`, then every `.invoke()` takes
`config={"configurable": {"thread_id": "..."}}`. Verified empirically: on
a SECOND call with the same `thread_id`, you only pass the NEW message --
LangGraph loads the checkpoint, merges your update on top via the normal
reducers, and every other field (`intent`, `decisions`, `audit_log`,
`total_tokens_used`, ...) is exactly where the last turn left it. A
DIFFERENT `thread_id` is a completely fresh, isolated state -- verified
that too. `MemorySaver` is in-process/dev-only (lost on restart); a real
deployment swaps in `SqliteSaver`/`PostgresSaver` -- changes one line,
nothing about the graph itself.

### A real bug checkpointing exposed, not just a feature it enabled
`agent_turns` (Milestone 9) was an `operator.add` reducer. Fine when every
`.invoke()` started blank -- Milestone 11 made state PERSIST, and the
reducer kept accumulating ACROSS separate turns instead of resetting.
Verified live on the real graph: turn 1 ended at `agent_turns=1`, turn 2
ended at `3`, not reset. Root cause: `decisions`/`audit_log` (also
`operator.add`) are CORRECTLY thread-lifetime-scoped -- a growing audit
trail across a whole conversation is exactly right. `agent_turns` is
different: its whole PURPOSE is bounding one turn's tool-calling loop, not
the conversation. Not every reducer field shares the same correct scope;
each has to be reasoned about on its own semantics, not copy-pasted.

Fix: `agent_turns` is now plain (no reducer), explicitly reset to 0 every
turn in `redact_pii_node` (the first node any turn hits), and
`harness.build_tools_node` does `state.get("agent_turns", 0) + 1`
(read-then-increment) instead of relying on auto-accumulation. Verified
fixed on the same real scenario that exposed it: turn 2 correctly showed
`agent_turns=1`, not accumulated.

### Concept: interrupt() -- a REAL pause, not a simulated one
`interrupt(payload)` inside a node pauses the ENTIRE checkpointed run at
that exact point -- verified this works even when called from inside a
nested specialist subgraph (our real architecture), with the checkpointer
only on the top-level parent graph. The caller's `.invoke()` returns
immediately with `result["__interrupt__"]` describing what's pending,
instead of a normal final answer -- meaning any caller (our own demo
script, or eventually a real API) has to explicitly check for this and
handle it differently from a completed response. Resuming:
`app.invoke(Command(resume=<value>), config=same_thread)` -- `interrupt()`
returns that value instead of pausing again.

**Real gotcha, verified directly, not assumed:** on resume, the node
function RE-RUNS FROM THE TOP, not from the interrupt() call -- confirmed
by seeing the "about to call interrupt()" print statement fire a second
time before the "RESUMED with..." one. Any code BEFORE `interrupt()` in
that same node runs TWICE. Nothing in `human_escalation` does, but it's a
real bug source for any node this pattern gets copied into later.

**Real tradeoff, also caught live:** wiring in `interrupt()` broke an
existing test -- `human_escalation` could no longer be called as a bare
Python function outside a compiled graph (`RuntimeError: Called get_config
outside of a runnable context`). `interrupt()` needs LangGraph's runtime
context; a node using it stops being a pure, trivially-unit-testable
function the way `route_after_tools` etc. still are. Fixed by testing it
through an actual minimal graph instead (`test_checkpointing.py`) -- not
by avoiding the pattern, since the capability (a genuine pause point) is
worth the testability cost.

### Verified live, full system: memory AND human-in-the-loop together
Turn 1 established a name ("Dana") and got a risk grade. Turn 2, same
thread, asked only "What's my name?" -- answered correctly from persisted
history, zero new tool calls, compliance correctly skipped logging (no
real tool ran that turn -- the Milestone 8 fix holding up in a genuinely
new context). Separately: a real escalation (bad income data) paused the
graph with the exact expected interrupt payload, and resuming produced a
final message combining BOTH the automated escalation text and the
simulated human's actual input -- a real pause-and-resume, not a mocked one.

---

## Milestone 12 — MCP: real tool transport

**Files:** `mcp_server.py` (new), `mcp_client.py` (new), `harness.py`
(rewritten), all four specialists, `graph.py`, `skill_evals.py`

### Concept: MCP server/client, and what actually changed
`mcp_server.py` wraps the SAME `tools/*.py` functions (`get_fraud_score`,
`get_risk_grade`, `get_claim_severity`, `flag_for_fraud_review`,
`search_policy_docs`) behind `@mcp.tool()` -- the mock logic is byte-for-
byte unchanged, verified directly (same fraud_score/risk_band/drivers
through MCP as through direct import). `mcp_client.py` fetches tool
objects from that server (launched as a stdio subprocess) once at each
specialist's module-import time -- same "build once, reuse every
request" pattern as `llm.py`'s client, just for tools instead of the LLM.
This is the blueprint's "swap mock for real model, zero agent-code
changes" premise made LITERAL for the first time: the swap will happen
entirely inside `mcp_server.py`/`tools/*.py`, and no specialist file will
need to change at all.

### Concept: MCP tools are async-only, and that propagates further than it looks
Verified directly: `.invoke()` on an MCP-sourced tool raises
`NotImplementedError`. Only `.ainvoke()` works. Also verified: LangGraph
requires the WHOLE containing graph to run via `.ainvoke()` the moment
ANY node is async -- no silent bridging -- and this holds even through
NESTED subgraphs (every specialist here, nested in the parent
orchestrator). So one fact ("MCP tools need await") forced `.ainvoke()`
everywhere a graph gets invoked: every specialist's own `__main__`,
`graph.py`'s `__main__`, `skill_evals.py`'s live runner. Also verified,
which kept the blast radius smaller than it could have been: a graph
mixing plain sync nodes and async nodes runs fine under `.ainvoke()` --
only `harness.py`'s tools node itself needed to become `async def`;
`call_model`, `classify_intent`, `summarize_decision`, `redact_pii_node`
all stayed exactly as they were.

**Interview framing:** "does adding one async dependency force a rewrite
of everything?" -- no, verified specifically: async-ness propagates
through *invocation* (whoever calls `.invoke()` vs `.ainvoke()`), not
through every node's own implementation. Knowing that distinction is what
keeps a real migration like this from being a full rewrite.

### Real bug #1: the subprocess can't find its own package
First attempt at `mcp_client.py` failed immediately:
`ModuleNotFoundError: No module named 'argus'` -- the subprocess running
`mcp_server.py` doesn't inherit the parent process's `PYTHONPATH`.
Fixed by having the SERVER make itself self-sufficient
(`sys.path.insert(0, ...)` at its own top, before any `argus` import)
rather than depending on however it happens to be launched -- matters
again once this runs under Docker/CI later, where the launching
environment won't look like a local dev shell at all.

### Real bug #2, the significant one: MCP swallows tool exceptions
First working version used a plain `try/except` around `tool.ainvoke(args)`
-- worked fine for the happy path, but a live human-in-the-loop test that
used to pause correctly (Milestone 11) silently STOPPED pausing once MCP
was wired in. Root cause, found by isolating a minimal failing MCP tool
and inspecting exactly what comes back: **FastMCP catches a tool's raised
exception SERVER-SIDE and returns a normal-looking, non-exceptional
result whose text happens to describe the error.** `langchain-mcp-
adapters` converts this into a `ToolException` internally, but that too
gets caught by its own `handle_tool_error` callback and turned back into
ordinary content -- so `.ainvoke()` never raises at all for a genuine
tool-side error. A `try/except` around it is dead code for this failure
mode.

The only reliable signal: `ToolMessage.status` (`"success"`/`"error"`) --
but only available if you invoke with the FULL tool-call dict
(`{"name", "args", "id", "type": "tool_call"}`, exactly what
`response.tool_calls` already gives you), not just the bare args. Verified
both shapes directly: bare args -> raw content, no status at all; full
call dict -> a proper `ToolMessage` with a real status field.

**Then a second twist, also verified rather than assumed:** a PLAIN
(non-MCP) tool with no `handle_tool_error` configured behaves the
OPPOSITE way -- it raises a real Python exception through `.ainvoke()`,
it does NOT swallow it into a status field. So the harness needed BOTH
mechanisms together, not either alone: `try/except` for plain tools and
genuine transport/connection failures, AND a `.status` check for MCP's
internally-caught tool errors. `_invoke_with_retry` in `harness.py` now
does both. Re-verified against the exact scenario that broke: escalation
correctly paused again, with the retry log correctly showing 2 attempts.

**Interview framing:** "the tool layer failed and my try/except didn't
catch it" is a completely realistic production incident -- different
tool transports (MCP, a REST wrapper, a message queue) can each have
their OWN way of representing failure that isn't necessarily "raise a
Python exception," and a harness built assuming only one failure shape
will have blind spots exactly like this one, silently, until something
forces it to surface (here: a test that specifically checked whether the
graph paused, not just whether it "finished without crashing").

### Real, live-observed operational cost, not just a design tradeoff
Full test suite went from ~4s to ~22s once specialist modules started
fetching MCP tools (spinning up a real subprocess) at IMPORT time --
every test file importing any specialist now pays that startup cost.
Named as a real, felt tradeoff of the "fetch once at module load" pattern
applied to something with actual subprocess-launch latency, not just a
cheap LLM client construction.

---

## Milestone 13 — FastAPI service wrapper

**Files:** `src/argus/api.py` (new), `tests/test_api.py` (new)

### The open question from Milestone 12, resolved by actually running it
`mcp_client.py`'s docstring flagged a real risk: `asyncio.run()` at each
specialist's import time might not survive being imported by an async web
framework. Resolved empirically, not by reasoning about it in the
abstract -- built the real FastAPI app, ran it via actual `uvicorn`,
hit it with real HTTP requests. It worked. The reason: `from argus.graph
import build_graph` sits at TRUE top-level module scope in `api.py`, which
Python executes synchronously during process startup, BEFORE uvicorn's
event loop exists -- exactly like a plain script or pytest collection.
**The risk was real, but only for a different pattern** -- deferring that
same import into an async `lifespan`/`startup` hook, which genuinely
would hit "cannot run event loop while another is running." Verified
which side of that line the working code is on, not assumed.

### Concept: two endpoints, one shared response-shaping function
`POST /chat` (start or continue a conversation) and `POST /resume`
(continue a PAUSED one) both funnel through `_build_response()`, which
branches on `"__interrupt__" in result` -- reusing `extract_text()` from
Milestone 10 so Gemini's content-block shape never leaks into the JSON
API response. `/resume` validates there's actually something to resume
via `(await app_graph.aget_state(config)).next` -- verified directly:
`()` (empty tuple) when nothing's paused, a non-empty tuple naming the
paused node otherwise. Returns a real `400` instead of a confusing result
for a thread with nothing pending.

### A subtle, verified finding about paused NESTED subgraphs
Live-tested the full escalation → resume flow over real HTTP and noticed
something worth chasing rather than shrugging off: the paused response's
`agent_turns` read `0`, while the SAME request's `interrupt_payload.agent_turns`
(built from inside `human_escalation`) read `1`. Reproduced with a clean
minimal example to confirm rather than guess: a child subgraph's OWN
earlier, already-completed node updates are invisible to the PARENT
graph's `ainvoke()` return value until the child subgraph itself finishes
and returns control -- while it's paused mid-way, the parent has received
NOTHING from it yet, even progress made before the pause. Confirms
`human_escalation`'s design (baking everything relevant directly into the
`interrupt()` payload, Milestone 11) isn't just convenient -- it's
necessary, since the parent's own state genuinely doesn't have that
information available while paused.

**Interview framing:** "what does the caller see while a nested subgraph
is paused" is a sharp, specific question about how LangGraph's
parent/child state boundary interacts with `interrupt()` -- the answer is
"only what completed BEFORE the paused node started, from the parent's
point of view," which is exactly why anything the caller needs to know
about a pending interruption has to be explicitly included in the
`interrupt()` payload itself, not assumed recoverable from state.

---

## Milestone 14 — Eval/test maturity for CI

**Files:** `src/argus/rag_evals.py` (new), `src/argus/eval_gate.py` (new),
`tests/test_rag_evals.py`, `tests/test_eval_gate.py`

### Concept: two-tier CI, on purpose -- pytest vs. the eval gate
`pytest` covers cheap, deterministic logic and should run on EVERY push
(seconds, zero LLM cost). `eval_gate.py` covers expensive, LLM-dependent
QUALITY regressions -- a prompt getting worse, retrieval missing
documents, an answer drifting ungrounded -- meant to gate a merge or run
nightly, not fire on every commit. Deliberately kept as two separate
things, not merged into one pytest run, because they check fundamentally
different kinds of regression at fundamentally different costs.

### Concept: hand-built RAGAS-style metrics, same philosophy as rag.py itself
`rag_evals.py` doesn't pull in the real RAGAS library -- same "build it
by hand so the mechanics are visible" choice as `rag.py`'s own cosine
similarity. Two metrics:
- `context_precision`: a simplified hit-rate@k against 8 hand-labeled
  (query, expected_doc_id) pairs, one per corpus document. Real
  embedding calls, no generation -- cheap enough for pytest. Named
  honestly as a SIMPLIFICATION of real RAGAS context precision (which
  also weighs ranking position, not just presence in top-k).
- `judge_answer` (faithfulness + relevance): LLM-as-judge -- a SEPARATE
  model call scores a generated answer against what was actually
  retrieved, rather than trusting the generating model's own
  self-assessment. Requires a full live agent round-trip -- live-only,
  never in pytest.

### Real bug, caught immediately, and recognized on sight
First version of `eval_gate.py` deferred the specialist imports
(`build_fraud_agent` etc.) into `_run_live_checks()`, an async function
run via `asyncio.run()`. Crashed instantly: `RuntimeError: asyncio.run()
cannot be called from a running event loop` -- because those specialist
modules call `get_mcp_tools()` (Milestone 12) at THEIR OWN import time,
which does its own internal `asyncio.run()`. A nested `asyncio.run()`
inside an already-running one. This is the EXACT trap `api.py`
(Milestone 13) already had to route around by importing at true top-level
-- missed applying the same fix here the first time, recognized the
error message immediately because it was the one predicted in
`mcp_client.py`'s own docstring. Fixed by moving those imports to real
module top-level in `eval_gate.py`, same as `api.py`. A pattern learned
once still has to be actively RE-APPLIED at each new call site -- it
doesn't propagate on its own.

### The thing that mattered most, verified rather than assumed
A CI gate that can only ever report PASS is worse than no gate at all --
a false sense of safety. `test_eval_gate.py` exists specifically to
prove the FAIL path works, not just the happy path: monkeypatches a
single failing offline eval case, a below-threshold context_precision,
and a failed live-check batch, and confirms `main()` returns `1` in each
case, not silently `0`. Verified live too: the real gate run (real LLM
calls, real MCP tools, real retrieval) exited `0` with every check
genuinely passing, including the LLM-as-judge correctly identifying a
real Policy answer as faithful and relevant with a concrete explanation,
not a rubber-stamp.

---

## Milestone 15+16 — Dockerize + CI/CD (pulled forward together)

**Files:** `Dockerfile`, `.dockerignore`, `.github/workflows/ci.yml`,
`.github/workflows/eval-gate.yml`, `mcp_client.py` (fixed)

Landed together because verifying Milestone 15's Dockerfile genuinely
required Milestone 16's CI (no local Docker in the dev sandbox) --
building the eval-gate/CI split first, then Dockerizing on top of it,
would have meant redoing the verification path twice.

### Environment constraint, handled by asking rather than working around solo
No Docker/Podman available in this sandbox, no passwordless sudo to
install one. First instinct was to build an elaborate local simulation
(fresh venv, `env -i`, a mirrored directory) to verify as much as
possible anyway -- corrected mid-stream: should have stopped and asked
how to proceed as soon as the constraint was clear, not silently kept
running tool calls solo. Course-corrected on direct feedback.

### A real bug the simulation surfaced anyway -- worth having built it
Despite the detour, the simulation (a directory with only what the
Dockerfile's COPY steps bring in, zero `.env` file anywhere, API key
injected purely as an environment variable) genuinely reproduced a real
container-relevant bug: the app crashed with `GOOGLE_API_KEY not set`
INSIDE the MCP server subprocess, even though the parent process had it.

**Root cause, found by reading the actual library source, not guessed:**
`mcp/client/stdio/__init__.py`'s `get_default_environment()` is a
deliberate SECURITY ALLOWLIST -- the stdio launcher does not inherit the
parent process's full environment. On Linux it passes through exactly six
variables (`HOME`, `LOGNAME`, `PATH`, `SHELL`, `TERM`, `USER`) and nothing
else, by design: an MCP server may be a third-party, less-trusted
process, and blindly forwarding secrets to every tool server you launch
would be a real security hole.

**This reframes Milestone 12's PYTHONPATH fix, not just explains a new
bug.** The `sys.path.insert()` fix in `mcp_server.py` wasn't fixing an
isolated subprocess quirk -- it was working around this exact allowlist,
just via a different route (making the server self-sufficient instead of
passing the variable through). Never connected the two until this
milestone's testing forced the allowlist itself into view. Every earlier
test had a real `.env` file on disk, which the subprocess's OWN
`load_dotenv()` call found independently -- masking that env-var
inheritance was never happening at all, the whole time.

**Fix:** `StdioServerParameters` accepts an `env` field that MERGES on
top of the safe default set (verified by reading the exact merge line:
`{**get_default_environment(), **server.env}`) -- not a full override.
`mcp_client.py` now passes `GOOGLE_API_KEY` through explicitly via this
mechanism. Deliberately did NOT pass `os.environ` wholesale, which would
defeat the entire point of the allowlist -- pass only what's actually
needed, preserving the security property rather than working around it.
Re-verified against the exact scenario that broke: same isolated,
no-`.env` environment, now boots cleanly and correctly answers a real
`search_policy_docs` question (the tool whose subprocess needs the key).

### Resolved: real Docker verification, via CI (no local Docker available)
No Docker/Podman in the dev sandbox, no passwordless sudo to install one.
Corrected mid-stream on direct feedback: was building an elaborate local
workaround (fresh venv, `env -i`) without checking in on the approach
first -- stopped and asked. Resolution: pushed to a real GitHub repo
(`waqar9425/bfsi-genai-project`) and let GitHub Actions' hosted runners
(real Docker, preinstalled) do the actual `docker build`/`docker run` --
arguably more industry-authentic anyway (professional teams verify Docker
builds in CI, not by hand on a laptop).

**First real CI run, immediately useful despite "failing":** `docker
build` and `pip install` both SUCCEEDED on a genuinely clean machine
(real, first-ever confirmation the Dockerfile is correct) -- the failures
were `pytest` and the smoke test, both because `GOOGLE_API_KEY` wasn't
yet set as a repo secret. Expected, informative, not a real problem.

**Second real run, after the secret was added, surfaced two ACTUAL
issues -- both fixed, from real failure output, not reasoning in the
abstract:**
1. The smoke test asserted `"paused":false` -- a specific BUSINESS
   outcome. A real transient tool failure made the harness correctly
   escalate instead of crash (exactly Milestone 4's design, working as
   intended) -- and the smoke test failed anyway, because it was
   checking the wrong thing. A smoke test should assert the deployment
   is well-formed and responsive, not that every downstream LLM/tool
   call succeeds at that exact moment. Fixed: assert `thread_id`/`reply`
   presence, accept either a normal answer or a valid escalation as pass.
2. `eval-gate.yml`'s `push: branches: [main]` trigger fired CONCURRENTLY
   with `ci.yml` on the same push -- both hammering the same free-tier
   key at once, directly causing the transient failure above. Removed;
   manual dispatch + nightly only, which is also a more honest read of
   "merge-time/nightly, not every commit" than "every push" ever was.

### Two-tier CI, materialized for real
`ci.yml` (fast: pytest + docker build + smoke test, every push/PR) and
`eval-gate.yml` (slow: `eval_gate.py`, manual + nightly) are the literal
GitHub Actions realization of the two-tier design `eval_gate.py` itself
was built around back in Milestone 14 -- not a new design made for CI,
the CI is what that design was always for.

### A genuinely valuable side effect of not having Docker locally
Testing via a hand-built local simulation (before pivoting to real CI)
still surfaced a real, container-relevant bug on its own: the MCP SDK's
stdio launcher does NOT inherit the parent process's environment --
`mcp/client/stdio/__init__.py`'s `get_default_environment()` is a
deliberate security allowlist (`HOME`/`LOGNAME`/`PATH`/`SHELL`/`TERM`/
`USER` only on Linux), root-caused by reading the actual SDK source, not
guessed. This retroactively explains Milestone 12's `PYTHONPATH` fix too
-- same root cause, never connected until now, masked every earlier time
by a real `.env` file always being on disk. Fixed by passing
`GOOGLE_API_KEY` explicitly via `StdioServerParameters`' `env` field
(merges on top of the safe defaults) -- deliberately not passing
`os.environ` wholesale, which would defeat the allowlist's purpose.

### Status
Second CI run (with the two fixes above) pushed but not yet confirmed --
hit GitHub's unauthenticated API rate limit (60 req/hour) from a
polling-loop mistake, ~6 hour wait before checkable again from here.
Real, live-observed lesson of its own: an unauthenticated polling loop
against a REST API is a genuinely bad pattern -- should have used a
single longer-interval check or just pointed at the Actions UI directly,
not repeated tight polling.

---

## Phase D — pivot to classical ML models
This closes out the planned agentic-build roadmap (Milestones 0-16).
Next: swap the mocked tools (`tools/*.py`'s rule-based logic) for real
trained models -- fraud (XGBoost + Isolation Forest), risk grading
(Logistic Regression -> XGBoost), claims severity (Gradient Boosting
Regressor), per the blueprint's Section 07. Because every tool has lived
behind a fixed, typed contract since Milestone 2 (and behind a real MCP
server since Milestone 12), this swap should touch ONLY `tools/*.py` and
`mcp_server.py` -- zero changes to any specialist, any prompt, any graph
wiring. The blueprint's core premise, about to be tested for real.
