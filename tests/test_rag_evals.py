"""
context_precision makes real embedding calls (via the already-built
corpus index in policy_tools.py) but no generation -- cheap enough for
pytest, unlike judge_answer (a full agent round-trip + a judge call),
which stays live-only in eval_gate.py.
"""

from argus.rag_evals import RETRIEVAL_EVAL_CASES, context_precision


def test_retrieval_hit_rate_meets_threshold():
    precision, details = context_precision()
    failures = [d for d in details if not d["hit"]]
    assert precision >= 0.85, f"hit rate {precision:.2f} below threshold, misses: {failures}"


def test_every_corpus_doc_has_a_labeled_case():
    # Sanity check on the eval set itself -- if policy_corpus.py gains a
    # doc with no case here, retrieval could silently regress on that
    # topic and nothing would ever test it.
    from argus.tools.policy_corpus import POLICY_DOCS

    labeled_ids = {c.expected_doc_id for c in RETRIEVAL_EVAL_CASES}
    corpus_ids = {d["doc_id"] for d in POLICY_DOCS}
    assert labeled_ids == corpus_ids
