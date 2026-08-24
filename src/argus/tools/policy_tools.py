"""
Real RAG tool: search_policy_docs. Unlike get_fraud_score/get_risk_grade,
this tool doesn't return a definitive answer the model just relays -- it
returns raw, ranked excerpts the model must READ, judge the relevance of,
and synthesize an answer from, citing which doc_id it drew from. See
Lesson 6 in LEARNING_NOTES.md for why that's a meaningfully different kind
of tool than every other one in this codebase.
"""

from langchain_core.tools import tool

from argus.rag import VectorStore
from argus.tools.policy_corpus import POLICY_DOCS

# Copy each dict -- VectorStore attaches an "_embedding" key to whatever
# it's given, in place. Don't mutate the module-level POLICY_DOCS list.
_store = VectorStore([dict(d) for d in POLICY_DOCS])


@tool
def search_policy_docs(query: str) -> list[dict]:
    """Search policy wordings, coverage rules, claims process, and KYC
    documentation for information relevant to a customer's question.

    Args:
        query: What to search for, in the customer's own terms.

    Returns the top matching document excerpts with their doc_id and a
    relevance score. Always cite the doc_id when answering from these
    results. If nothing relevant comes back, say so plainly -- do not
    guess or answer from general insurance knowledge.
    """
    results = _store.search(query, k=3)
    return [
        {"doc_id": r["doc_id"], "text": r["text"], "relevance": round(r["score"], 3)}
        for r in results
    ]
