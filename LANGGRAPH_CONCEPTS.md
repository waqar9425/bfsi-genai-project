# LangGraph Concepts — Explained From Scratch

This file explains every LangGraph idea used in this project, assuming
zero prior LangGraph knowledge. If you know Google ADK, a few honest
comparisons are marked **[ADK contrast]** — take those as rough
orientation, not exact API mappings, since the two frameworks are built
on different philosophies.

**This file is updated after every milestone.** If a concept is used in
the code but you don't see it explained below, that's a bug in this file
-- ask for it to be added.

---

## 0. The one-sentence idea

A plain LLM call is a function: text in, text out, done. The moment an
agent needs to *loop* -- call a tool, look at the result, decide to call
another tool or finally answer -- something has to hold that loop's state
and decide what happens next. **LangGraph's answer: make the loop an
explicit graph.** Nodes are Python functions. Edges are transitions
between them (fixed, or decided by a function at runtime). A shared
`State` object flows through every node. That's the entire framework --
everything below is built from just these pieces.

**[ADK contrast]** ADK gives you pre-built orchestration shapes as
classes -- `SequentialAgent`, `ParallelAgent`, `LoopAgent` -- that hide the
control flow inside the class. LangGraph doesn't have those classes; you
build the equivalent shape yourself out of nodes and edges. More typing,
but nothing is hidden -- if you can't answer "what runs after this node
and why," that's a bug in your graph, not a framework mystery.

---

## 1. StateGraph, State, Nodes, Edges -- the absolute basics

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    count: int

def add_one(state: State) -> dict:
    return {"count": state["count"] + 1}   # <- a NODE: a plain function

g = StateGraph(State)          # start building a graph over this State shape
g.add_node("add_one", add_one) # register the node under a name
g.add_edge(START, "add_one")   # START -> add_one (every graph needs an entry edge)
g.add_edge("add_one", END)     # add_one -> END (every graph needs an exit)
app = g.compile()              # turn the definition into something runnable

app.invoke({"count": 0})       # -> {"count": 1}
```

- **State** is a schema (here a `TypedDict`) describing everything that
  flows through the graph.
- **A node** is a plain Python function: `(state) -> dict`. It receives
  the *whole* current state, and returns only the *keys it's updating* --
  not the whole state back. This matters a lot, see the reducers section.
- **An edge** connects two nodes. `START` and `END` are special markers
  for "where the graph begins" and "where it's allowed to stop."
- **`.compile()`** turns the graph definition into a runnable object with
  an `.invoke(initial_state)` method.

**[ADK contrast]** roughly: ADK's `Agent` (with instructions + tools) is
one unit of "an LLM that can act"; LangGraph's *node* is a much smaller
unit -- often just "one LLM call" or "one Python function." What ADK calls
one Agent frequently becomes several LangGraph nodes wired together (an
"agent" node, a "tools" node, maybe more) -- see Section 5.

---

## 2. State and reducers -- the single most important concept here

A node only returns the keys it's *changing*. LangGraph merges that
partial update back into the full State. The question is: **what does
"merge" mean when a key already has a value?**

By default: **overwrite** (last write wins). If you want something
different -- like "append to a list instead of replacing it" -- you
attach a **reducer**:

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # reducer: APPENDS, doesn't overwrite
    intent: str                               # no reducer: last write wins
```

Without `add_messages`, every node that touched `messages` would silently
**wipe conversation history**, because plain dict-merge overwrites by
default. `add_messages` is a specific reducer built for chat history: new
messages get appended to the existing list.

`intent` has no reducer on purpose -- only the *latest* routing decision
matters, we don't want a history of every intent ever assigned.

**Why this is the concept to really internalize:** every weird LangGraph
error you'll hit traces back to reducers. Concretely, we proved this in
this project -- two nodes writing the same *non-reducer* key in the same
parallel step doesn't silently pick one, **it crashes**:

```
InvalidUpdateError: At key 'intent': Can receive only one value per step.
Use an Annotated key to handle multiple values.
```

That's not a bug to work around -- it's LangGraph refusing to guess at an
ambiguous merge. If you want two things happening in parallel to both
write to the same field safely, that field needs a reducer (e.g.
`operator.add` for "combine these lists/numbers together").

**One more `add_messages` nuance (Milestone 9):** it doesn't only append --
if the message you return has the SAME `.id` as a message already in the
list, it REPLACES that message instead of appending a duplicate ("upsert
by id"). This is how PII redaction edits an already-in-state message: read
the existing message's id, build a replacement with that same id and
redacted content, return it. Verified empirically before relying on it --
confirmed the final message count stayed the same (no duplicate), content
was actually replaced.

**[ADK contrast]** ADK's session `state` dict is more free-form key-value
storage you read/write directly; there isn't a reducer concept -- you
decide merge behavior yourself, imperatively, every time you touch it.
LangGraph makes you declare the merge behavior once, up front, in the
schema -- which is what makes concurrent/parallel updates (Section 6)
safe instead of a footgun.

---

## 3. Conditional edges -- routing

A normal edge is fixed: `A` always goes to `B`. A **conditional edge**
picks the next node at runtime, based on state:

```python
def route_by_intent(state: State) -> str:
    return state["intent"]   # returns a NODE NAME, not a state update

g.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {"fraud": "fraud", "claims": "claims"},   # map returned string -> real node name
)
```

The router function is a plain `(state) -> str`. LangGraph calls it,
takes the returned string, looks it up in the mapping dict, and goes
there. If the router already returns real node names (or the special
`END` marker) directly, the mapping dict can be omitted entirely.

A genuinely useful, verified-in-this-project fact: **the mapping dict can
remap even a returned `END`** to a real node instead of actually ending
the graph -- used in Milestone 8 to insert a new step (`summarize`)
between "the agent is done" and "the subgraph actually ends," without
touching the router function that decided "the agent is done" at all.

**A router is a plain, pure function** -- test it directly, no graph
execution needed:
```python
assert route_by_intent({"intent": "fraud", ...}) == "fraud"
```

---

## 4. Tool calling -- two modes, and why they're actually the same mechanism

**Structured output** (`with_structured_output(SomeSchema)`): the model is
*forced* to return data matching your schema, every single call, no
exceptions, no free-text option. Used for the Orchestrator's routing
decision -- it must always produce an intent.

**Optional tool calling** (`bind_tools([tool1, tool2])`): the model is
*offered* tools but free to call zero, one, or several, or just reply in
plain text. Used by every specialist -- it might need to ask a clarifying
question instead of acting.

**The thing worth really knowing:** these are the same underlying wire
mechanism. Structured output is "tool calling, but there's only one fake
tool (your schema) and the model is forced to call it." When you see
"AFC" (automatic function calling) in Gemini's logs, that's this exact
mechanism, just named by the provider.

A tool is a plain Python function decorated with `@tool`:
```python
@tool
def get_fraud_score(claim_amount: float, prior_claims_count: int) -> dict:
    """Score a claim for fraud risk. <- this docstring IS the tool's
    description, exactly what the model reads to decide when to use it."""
    ...
```
The function's type hints become the schema the model sees; the docstring
becomes the description. A vague docstring is a vague API -- to the model,
same as it would be to a human reading your function signature.

**Gotcha (Milestone 9), verified empirically:** `with_structured_output(Schema)`
returns ONLY the parsed object by default -- there's no way to read token
usage, or anything else about the raw model response, from it. Pass
`include_raw=True` to get `{"raw": AIMessage, "parsed": Schema,
"parsing_error": ...}` instead. Side benefit: this ALSO changes what
happens when the model's output fails schema validation -- without
`include_raw`, that raises immediately; with it, you get the error object
handed to you (`parsing_error`) instead of a crash, and can decide what to
do about it yourself.

**[ADK contrast]** ADK's `FunctionTool` wrapping is conceptually the same
idea (a Python function becomes something the model can call), auto-derived
from the function signature similarly. The `bind_tools` vs
`with_structured_output` distinction (offered-but-optional vs
forced-single-schema) is more explicit/visible in LangGraph than it
typically is in ADK's higher-level agent config.

### The ReAct loop
Every specialist in this project is shaped like this:
```
   ┌───────┐   tool_calls present?   ┌────────┐
   │ agent │────────────────────────▶│  tools  │
   └───┬───┘◀────────────────────────└────────┘
       │ no tool_calls
       ▼
      END
```
Call the model → if it asked for a tool, run the tool(s), feed results
back → call the model again → repeat until it stops asking for tools →
done. This pattern has a name older than function-calling APIs: **ReAct**
(Reasoning + Acting, Yao et al. 2022). Originally it was free-text
"Thought: ... Action: ..." parsed with regex, because models had no
native tool-calling. Modern function-calling APIs are ReAct made a first-
class model capability -- what we built IS ReAct, just running on
structured calls instead of parsed text.

**Two prebuilt-equivalent pieces make this loop trivial:**
- A router checking "does the last message have tool_calls?" → routes to
  a tools-execution node or to `END`.
- A tools-execution node that runs whatever's in `.tool_calls` and appends
  the results as `ToolMessage`s.

In this project these are hand-written in `harness.py`
(`tools_present`, `build_tools_node`) rather than LangGraph's prebuilt
`tools_condition`/`ToolNode`, specifically so retry-and-escalation logic
(Section 7) could be layered in -- see PROJECT_MENTAL_MODEL.md for why.

---

## 5. Subgraphs -- a compiled graph IS a node

`StateGraph(...).compile()` returns something that satisfies the exact
same interface (`invoke`) that any node needs to satisfy. So you can
register a *whole other compiled graph* as a node in a bigger graph:

```python
fraud_agent = build_fraud_agent()       # a fully compiled ReAct loop, on its own
g.add_node("fraud", fraud_agent)        # ...used directly as ONE node here
```

This works with **zero special wrapper code** *when both graphs share the
exact same State schema* -- the parent's state flows in unmodified, the
child's updates merge back via the same reducers. This is the "easy case."
The moment a specialist needs *private* state the parent doesn't care
about, you'd need an explicit wrapper node translating state at the
boundary -- not built in this project yet, since it hasn't been needed.

**[ADK contrast]** this is roughly the LangGraph analogue of ADK's
`sub_agents` composition (an Agent delegating to other Agents) -- the
mechanism is different (a graph literally nested as a node vs. an agent
hierarchy), but the *idea* of "a bigger unit built out of smaller,
independently-defined units" is the same instinct.

---

## 6. Command -- a node deciding its own routing, dynamically

Every router so far is a **separate function**, declared ahead of time.
`Command` is different: the *node itself*, mid-execution, returns both a
state update AND the next hop, in one object:

```python
from langgraph.types import Command

def call_model(state) -> Command | dict:
    response = llm.invoke(...)
    if <model decided to hand off>:
        return Command(update={"messages": [response]}, goto="fraud")
    return {"messages": [response]}   # the normal case: just a state update
```

The genuinely powerful variant: `Command(goto="fraud", graph=Command.PARENT)`
lets a node **inside a subgraph jump to a node in the PARENT graph** --
something no ordinary conditional edge could ever do, since a subgraph's
own router only knows about its own nodes. This is how Claims Triage hands
a suspicious claim directly to Fraud Investigation mid-conversation,
bypassing the top-level router entirely for that turn.

**Real cost, not just a benefit:** a subgraph using `Command(graph=PARENT)`
now has an implicit dependency on a specific node existing one level up --
it's no longer fully self-contained or testable in isolation for that path.

---

## 7. Send -- dynamic parallel fan-out

A conditional-edge function can return a **list of `Send` objects**
instead of a single node-name string. Each `Send(node_name, state)` is an
independent, parallel invocation of that node, with its OWN local state:

```python
from langgraph.types import Send

def dispatch(state) -> list[Send]:
    return [Send("log_one", {"item": x}) for x in state["items"]]
```

This is LangGraph's "map" primitive -- N parallel branches from one node,
without hand-writing N edges. **Critical fact, proven in this project by
deliberately removing the reducer and watching it crash:** fan-out is only
safe because the field each branch writes to has a reducer. Remove it and
you get the *exact same* `InvalidUpdateError` from Section 2 -- Send-based
fan-out isn't a new failure mode, it's that same "concurrent writes need a
reducer" fact, now triggered on purpose.

---

## 8. recursion_limit -- the last-resort, graph-wide circuit breaker

Every graph has a default cap (25) on how many total steps it can take
before LangGraph forcibly stops it with `GraphRecursionError`. This is
independent of anything in your own code -- a safety net for the day a
loop you *thought* had an exit condition doesn't, for whatever reason.
Configurable per call: `app.invoke(state, config={"recursion_limit": N})`.

This is different from a hand-built retry budget (like the harness's
`MAX_ATTEMPTS`, see PROJECT_MENTAL_MODEL.md) -- that catches one specific,
known failure mode you designed for. `recursion_limit` catches *any*
runaway loop, for *any* reason, because LangGraph itself is watching the
step count, not your application logic.

---

## 9. Patterns built ON TOP of LangGraph (not LangGraph features themselves)

Worth being able to tell these apart from Sections 1-8 above, which are
all genuine LangGraph API surface:

- **The harness** (retry-then-escalate on tool failures, PLUS a
  request-wide turn budget as a second, independent reason to escalate --
  Milestone 9) -- ordinary nodes/edges/state, just organized as a reusable
  pattern.
- **PII redaction** (Milestone 9) -- a plain node using the `add_messages`
  upsert-by-id nuance above, applied at the graph's entry point.
- **Agent Skills** (Milestone 10) -- not a graph pattern at all, a prompt-
  engineering one: plain dataclasses holding reusable instruction text,
  imported into multiple specialists' system prompts instead of each
  hand-writing similar-but-drifting phrasing. Worth knowing this ISN'T a
  LangGraph feature if asked -- it's ordinary Python + string composition.
- **Reflection loops** (generate → critique → revise, bounded by an
  attempt counter) -- also just ordinary nodes/edges, no special API.
- **Compliance fan-out** (Section 7's `Send`, applied to audit logging) --
  a real application of a real LangGraph feature, not a new one.

None of these needed anything beyond Sections 1-7 above. That's
deliberate -- the point of learning LangGraph's actual primitives well is
that most "advanced" production patterns turn out to be ordinary graph
construction, not hidden framework magic.

---

## 10. Checkpointers -- persistence across separate .invoke() calls

```python
from langgraph.checkpoint.memory import MemorySaver
app = g.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "conversation-123"}}
app.invoke({"messages": [("user", "hi")]}, config=config)
app.invoke({"messages": [("user", "remember me?")]}, config=config)  # same thread -> sees BOTH messages
```

A `checkpointer` persists `State` after every super-step, keyed to a
`thread_id`. Verified empirically: on a second `.invoke()` with the same
`thread_id`, you only pass the NEW data -- LangGraph loads the saved
state, merges your update onto it via the normal reducers, and every
field is exactly where the last call left it. A different `thread_id` is
a completely isolated, fresh state -- verified that too.

`MemorySaver` is in-process, dev-only (lost when the process exits). A
real deployment uses `SqliteSaver`/`PostgresSaver` for actual durability
-- swapping which one changes ONE line (`g.compile(checkpointer=...)`),
nothing about the graph's nodes/edges/state schema changes at all.

**A real, live-caught lesson about reducers and persistence:** a field
that behaves fine when every run starts blank can be WRONG once state
persists across turns. This project had exactly that bug: a reducer field
meant to bound "how many tool-calling rounds has THIS turn taken" kept
accumulating across separate persisted turns instead of resetting,
because `operator.add` has no concept of "per-turn" vs "per-thread" --
it just keeps adding forever. Fixed by making it a plain field, explicitly
reset to 0 by the first node every turn hits, rather than a reducer.
Lesson: `decisions`/`audit_log`-style reducers (correctly thread-lifetime-
scoped -- a growing audit trail IS what you want) and turn-budget-style
reducers (incorrectly thread-lifetime-scoped by default) look identical
in the schema but need opposite treatment. Persistence forces you to
actually reason about each field's scope instead of copy-pasting a pattern.

## 11. interrupt() -- a genuine pause, mid-graph, waiting for a human

```python
from langgraph.types import interrupt, Command

def some_node(state):
    human_input = interrupt({"question": "approve this?"})  # PAUSES here
    return {"result": human_input}  # only reached after resume

# first call: PAUSES, returns immediately
result = app.invoke({...}, config=config)
result["__interrupt__"]  # -> [Interrupt(value={"question": "approve this?"}, id=...)]

# resume, same thread_id, later -- possibly a different process entirely
app.invoke(Command(resume="yes, approved"), config=config)
```

Requires a checkpointer (pausing/resuming needs somewhere to persist the
paused state). Verified this works even when the `interrupt()` call is
inside a node that's ITSELF a compiled subgraph nested inside a parent
(exactly this project's architecture -- Fraud/Underwriting/Claims/Policy
are all subgraphs-as-nodes) -- the checkpointer only needs to be on the
top-level parent graph.

**Two real gotchas, both verified directly rather than assumed:**
- **On resume, the node function re-runs from the top**, not from the
  `interrupt()` call -- confirmed by watching a print statement before
  `interrupt()` fire a SECOND time on resume. Any code with side effects
  before the `interrupt()` call in that same node executes twice.
- **A node calling `interrupt()` stops being a plain, directly-callable
  function.** It needs LangGraph's runtime context to work at all --
  calling it bare, outside a compiled graph's execution, raises
  `RuntimeError: Called get_config outside of a runnable context`. A real
  test in this project broke this exact way when `interrupt()` was added
  to a previously-pure node -- fixed by testing it through an actual
  minimal graph instead of calling the function directly. Worth knowing
  this tradeoff exists before reaching for `interrupt()`: you trade "pure
  function, trivially unit-testable" for "genuine pause capability."

**[ADK contrast]** ADK doesn't have a direct equivalent to this specific
mechanism as far as I'm confident about -- human-in-the-loop patterns
there tend to be built at the application layer (pausing/resuming a
session externally), rather than a graph-level primitive the framework
itself understands and can resume mid-node. Flagging this as a genuine
LangGraph-specific capability rather than guessing at an ADK mapping I'm
not sure of.

**One more nuance, added in Milestone 13, verified with a clean minimal
repro rather than assumed:** while a NESTED subgraph is paused mid-way,
the PARENT graph's `.ainvoke()` return value reflects only what completed
BEFORE the paused node started -- it does NOT include updates from
already-completed EARLIER nodes inside that same paused subgraph. Proven
directly: a child with `step1` (sets `count=1`) then `step2` (pauses)
showed `count=1` from INSIDE `step2` at the moment of pausing, but the
parent's `ainvoke()` return showed `count=0` -- `step1`'s update never
propagated up, because the subgraph-as-node hasn't "returned" to its
parent at all yet. Practical consequence: anything the caller needs to
know about a pending interruption MUST be explicitly included in the
`interrupt()` payload itself -- you cannot recover it from the parent's
own state while paused, even info that already existed inside the child.

## 12. Sync vs. async graphs -- the rules, verified directly

A node can be a plain `def` function OR an `async def` function.
Three rules, each verified empirically rather than assumed, all from
Milestone 12 (adopting MCP, whose tools are async-only):

1. **If ANY node in a graph is async, the graph must be invoked with
   `.ainvoke()`, never `.invoke()`.** No silent bridging -- calling
   `.invoke()` on a graph with an async node raises `TypeError: No
   synchronous function provided to "<node>"`, immediately, clearly.
2. **This holds through nested subgraphs.** A parent graph whose only
   async node is buried inside a child subgraph (itself a node in the
   parent -- exactly this project's architecture) STILL requires
   `.ainvoke()` at the parent level. Async-ness isn't contained by the
   subgraph boundary.
3. **But sync and async nodes coexist fine inside one graph, once you're
   using `.ainvoke()`.** A graph with three sync nodes and one async node
   runs correctly under `.ainvoke()` -- you do NOT need to convert every
   node to `async def` just because one of them has to be. Only whoever
   actually CALLS `.invoke()`/`.ainvoke()` needs to pick the right one;
   individual node implementations don't all need to match.

**Why this matters practically:** rule 3 is what kept a real MCP adoption
in this project from becoming a full-codebase rewrite. Only the harness's
tool-execution node needed `async def` (it's the only thing calling MCP
tools). Every LLM-calling node (`call_model`, `classify_intent`,
`summarize_decision`) stayed plain `def`, completely unmodified -- only
the handful of places that actually INVOKE a compiled graph
(`__main__` blocks, a future FastAPI route handler) needed to switch to
`await app.ainvoke(...)`.

**Interview question:** *"If I add one async tool to an otherwise-sync
LangGraph agent, what has to change?"* → Not the other nodes. Two things:
the node that calls the async tool becomes `async def`, and every call
site that invokes the graph (however many there are) switches from
`.invoke()` to `await ... .ainvoke()`. That's the whole blast radius,
verified directly rather than assumed -- rules 1-3 above are exactly what
determines it.

**[ADK contrast]** ADK's `Agent`/tool-calling machinery is async-native
throughout by default in newer versions, from what I'm aware of -- so this
sync/async boundary question is less likely to come up the same way there.
LangGraph's choice to support BOTH sync and async nodes, with an explicit
rule for what happens when you mix them, is a real design difference
worth naming if it comes up.

**Milestone 14 addendum: knowing a rule once isn't the same as applying
it everywhere.** The exact "don't defer a heavy import into an async
function" trap from this section bit `eval_gate.py` for real, immediately
-- `RuntimeError: asyncio.run() cannot be called from a running event
loop`, the precise error this section already documented, in a NEW file
that hadn't been written yet when the lesson was first learned in
`api.py`. Fixed the same way: move the import to true top-level. Worth
internalizing as a general truth, not just about this one rule -- a
pattern learned at one call site doesn't automatically apply itself at
the next one; it has to be actively re-applied, every time.
