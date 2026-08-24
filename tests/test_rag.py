"""
cosine_similarity is pure math -- test it directly, no embedding API call
needed. VectorStore.search itself isn't unit-tested here (it requires a
real embedding call per instantiation); it's exercised live via
policy_customer.py's __main__ instead, same tradeoff as other milestones
where the deterministic logic is unit-tested and the LLM/API-dependent
path is a live smoke test.
"""

from argus.rag import cosine_similarity


def test_identical_vectors_similarity_is_one():
    v = [1.0, 2.0, 3.0]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_orthogonal_vectors_similarity_is_zero():
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_opposite_vectors_similarity_is_negative_one():
    assert abs(cosine_similarity([1.0, 0.0], [-1.0, 0.0]) - (-1.0)) < 1e-9


def test_similarity_ignores_magnitude_not_just_direction():
    # [1,0] and [5,0] point the exact same direction -- cosine similarity
    # should be 1.0 regardless of the second vector being 5x longer. This
    # is the whole point of using cosine over raw distance.
    assert abs(cosine_similarity([1.0, 0.0], [5.0, 0.0]) - 1.0) < 1e-9


def test_zero_vector_does_not_divide_by_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
