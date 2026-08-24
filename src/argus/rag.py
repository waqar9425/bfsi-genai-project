"""
Minimal RAG (retrieval-augmented generation), built by hand rather than
through a vector-database library -- the point is to make the actual
mechanics visible. A real vector database (FAISS, pgvector, Pinecone) does
the exact same comparison at scale, using approximate-nearest-neighbor
indexing instead of brute-force comparison against every stored vector --
what it buys you is SCALE, not different math.
"""

import math
import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

_EMBEDDING_MODEL = "models/gemini-embedding-001"


def _get_embeddings_client() -> GoogleGenerativeAIEmbeddings:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set. Add it to a .env file at the project root.")
    return GoogleGenerativeAIEmbeddings(model=_EMBEDDING_MODEL, google_api_key=api_key)


_embeddings = _get_embeddings_client()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """The angle between two vectors, not their distance -- meaning
    correlates with DIRECTION in embedding space, not magnitude. Pure
    math, no API call, fully unit-testable on its own (see test_rag.py).
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    """The whole mechanic of a vector database, minus the scale: a list of
    (text, embedding) pairs, and brute-force cosine similarity to rank
    them against a query.
    """

    def __init__(self, documents: list[dict]):
        self._docs = documents
        texts = [d["text"] for d in documents]
        # One batched embedding call for the whole corpus, not N separate
        # calls -- embedding APIs support batching specifically so you
        # don't pay N round-trips to index N documents.
        vectors = _embeddings.embed_documents(texts)
        for doc, vec in zip(self._docs, vectors):
            doc["_embedding"] = vec

    def search(self, query: str, k: int = 3) -> list[dict]:
        query_vec = _embeddings.embed_query(query)
        scored = [
            {
                **{key: val for key, val in doc.items() if key != "_embedding"},
                "score": cosine_similarity(query_vec, doc["_embedding"]),
            }
            for doc in self._docs
        ]
        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:k]
