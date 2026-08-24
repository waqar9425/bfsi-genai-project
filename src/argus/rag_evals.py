"""
Milestone 14: RAGAS-style evals for Policy's retrieval -- lightweight,
hand-built versions of the standard RAG eval metrics, same "mocks before
infrastructure" philosophy as rag.py's own hand-built cosine similarity
(not the real RAGAS library, the core IDEAS it measures).

Two metrics, two different costs, same offline/live split as everywhere
else in this project:
- context_precision: does retrieval return the right document? Real
  embedding calls but no generation -- cheap enough for pytest
  (test_rag_evals.py uses it directly).
- judge_answer (faithfulness + relevance): LLM-as-judge, requires a full
  live round-trip through the real Policy agent AND a separate judge
  call -- live-only, exercised by eval_gate.py, never in pytest.
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field

from argus.llm import get_llm
from argus.tools.policy_tools import _store  # reuse the already-built corpus index


@dataclass
class RetrievalEvalCase:
    query: str
    expected_doc_id: str


# One hand-labeled case per document in policy_corpus.py -- if retrieval
# breaks for any one topic, this pinpoints which.
RETRIEVAL_EVAL_CASES = [
    RetrievalEvalCase("Is water damage from a burst pipe covered?", "COV-WATER-01"),
    RetrievalEvalCase("What's my auto collision deductible?", "COV-AUTO-DED-01"),
    RetrievalEvalCase("How do I file a claim?", "CLAIMS-PROCESS-01"),
    RetrievalEvalCase("Can I cancel my policy and get a refund?", "CANCEL-POLICY-01"),
    RetrievalEvalCase("Is my jewelry covered if it's stolen?", "COV-THEFT-01"),
    RetrievalEvalCase("What ID do I need to verify my identity?", "KYC-VERIFY-01"),
    RetrievalEvalCase("Does insurance cover wildfire damage?", "COV-FIRE-01"),
    RetrievalEvalCase("Can I pay my premium quarterly?", "PREMIUM-PAYMENT-01"),
]


def context_precision(k: int = 3) -> tuple[float, list[dict]]:
    """Simplified "context precision": for each labeled case, does the
    expected doc_id appear ANYWHERE in the top-k retrieved results?
    (Real RAGAS context precision also weighs WHERE in the ranking a hit
    lands -- this is the hit-rate@k simplification, named honestly, not
    the full metric.) Returns (hit_rate, per_case_detail).
    """
    details = []
    hits = 0
    for case in RETRIEVAL_EVAL_CASES:
        results = _store.search(case.query, k=k)
        retrieved_ids = [r["doc_id"] for r in results]
        hit = case.expected_doc_id in retrieved_ids
        hits += hit
        details.append(
            {"query": case.query, "expected": case.expected_doc_id, "retrieved": retrieved_ids, "hit": hit}
        )
    return hits / len(RETRIEVAL_EVAL_CASES), details


class FaithfulnessJudgment(BaseModel):
    faithful: bool = Field(
        description="True if every claim in the answer is supported by the retrieved context -- no invented facts"
    )
    relevant: bool = Field(description="True if the answer actually addresses the question asked")
    reasoning: str = Field(description="One sentence explaining the judgment")


def judge_answer(question: str, retrieved_context: str, answer: str) -> FaithfulnessJudgment:
    """LLM-as-judge -- a SEPARATE call from whatever generated the answer,
    scoring it against the retrieved context rather than trusting the
    generating model's own self-assessment. Live-only.
    """
    judge_llm = get_llm().with_structured_output(FaithfulnessJudgment)
    prompt = (
        f"Question: {question}\n\n"
        f"Retrieved context the answer should be grounded in:\n{retrieved_context}\n\n"
        f"Answer given: {answer}\n\n"
        "Judge whether the answer is FAITHFUL (no claims beyond what the "
        "context supports) and RELEVANT (actually addresses the question)."
    )
    return judge_llm.invoke([("user", prompt)])
