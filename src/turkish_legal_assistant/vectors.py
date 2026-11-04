"""Dense, sparse and hybrid retrieval with BGE-M3 embeddings in Qdrant.

Build the embedding cache once (downloads BGE-M3, ~2.3 GB, on first run):
    uv run python -m turkish_legal_assistant.vectors
"""

import json
from functools import cache

import numpy as np
from qdrant_client import QdrantClient, models

from .data import ARTICLES
from .ingest import DATA_DIR
from .models import Article

MODEL_NAME = "BAAI/bge-m3"
COLLECTION = "articles"
DENSE_FILE = DATA_DIR / "embeddings_dense.npy"
SPARSE_FILE = DATA_DIR / "embeddings_sparse.json"
PREFETCH = 50  # candidates each side contributes before hybrid fusion

Sparse = dict[int, float]  # token id -> weight


@cache
def get_model():
    from FlagEmbedding import BGEM3FlagModel  # heavy import (torch), only when embedding is needed

    return BGEM3FlagModel(MODEL_NAME, use_fp16=False)


def embed(texts: list[str]) -> tuple[np.ndarray, list[Sparse]]:
    """BGE-M3 gives both outputs in one pass: a dense meaning vector and sparse word weights."""
    # Small batches: attention memory grows with batch x length^2, and the default batch of 256
    # with 4096-token articles used 26 GB. Longest article is ~3300 tokens, so nothing is cut.
    out = get_model().encode(texts, batch_size=2, max_length=4096, return_dense=True, return_sparse=True)
    sparse = [{int(t): float(w) for t, w in weights.items()} for weights in out["lexical_weights"]]
    return out["dense_vecs"], sparse


def article_text(a: Article) -> str:
    # Law name, section and title give the chunk context its body text alone lacks (e.g. "1. Genel olarak").
    return f"{a.law_name}, madde {a.article_number}: {a.title}\n{a.section}\n{a.text}"


def _sparse_vector(s: Sparse) -> models.SparseVector:
    return models.SparseVector(indices=list(s), values=list(s.values()))


class VectorIndex:
    def __init__(self, articles: list[Article], dense: np.ndarray, sparse: list[Sparse]):
        assert len(articles) == len(dense) == len(sparse), "embedding cache does not match articles"
        self.articles = articles
        # In-memory Qdrant: 348 articles load in milliseconds and no file lock is shared between processes.
        # shortcut: rebuilt on every start; switch to QdrantClient(url=...) with a server in Phase 8.
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            COLLECTION,
            vectors_config={"dense": models.VectorParams(size=dense.shape[1], distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        self.client.upsert(
            COLLECTION,
            points=[
                models.PointStruct(
                    id=i,
                    vector={"dense": dense[i].tolist(), "sparse": _sparse_vector(sparse[i])},
                    payload={"law_number": a.law_number},
                )
                for i, a in enumerate(articles)
            ],
        )

    def search(
        self, query: str, k: int, mode: str = "hybrid", law_number: str | None = None
    ) -> list[tuple[Article, float]]:
        dense, sparse = embed([query])
        dense_q, sparse_q = dense[0].tolist(), _sparse_vector(sparse[0])
        flt = (
            models.Filter(must=[models.FieldCondition(key="law_number", match=models.MatchValue(value=law_number))])
            if law_number
            else None
        )
        if mode == "dense":
            kwargs = dict(query=dense_q, using="dense", query_filter=flt)
        elif mode == "sparse":
            kwargs = dict(query=sparse_q, using="sparse", query_filter=flt)
        elif mode == "hybrid":
            # Reciprocal Rank Fusion: each list votes 1/(60 + rank); scores of different scales never mix.
            kwargs = dict(
                prefetch=[
                    models.Prefetch(query=dense_q, using="dense", limit=PREFETCH, filter=flt),
                    models.Prefetch(query=sparse_q, using="sparse", limit=PREFETCH, filter=flt),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        hits = self.client.query_points(COLLECTION, limit=k, **kwargs).points
        return [(self.articles[h.id], h.score) for h in hits]


@cache
def get_index() -> VectorIndex:
    if not DENSE_FILE.exists():
        raise FileNotFoundError("Embedding cache missing: run `uv run python -m turkish_legal_assistant.vectors`")
    sparse = [{int(t): w for t, w in s.items()} for s in json.loads(SPARSE_FILE.read_text())]
    return VectorIndex(ARTICLES, np.load(DENSE_FILE), sparse)


def main() -> None:
    dense, sparse = embed([article_text(a) for a in ARTICLES])
    np.save(DENSE_FILE, dense)
    SPARSE_FILE.write_text(json.dumps(sparse))
    print(f"Embedded {len(ARTICLES)} articles: dense {dense.shape}, saved to {DENSE_FILE.parent}")


if __name__ == "__main__":
    main()
