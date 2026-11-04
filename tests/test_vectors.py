import numpy as np
import pytest

from turkish_legal_assistant import vectors
from turkish_legal_assistant.data import find_article

# Three real articles with hand-made 3-d "embeddings": each points along its own axis.
ARTICLES = [find_article("4857", "53"), find_article("6698", "11"), find_article("6098", "342")]
DENSE = np.eye(3, dtype=np.float32)
SPARSE = [{1: 1.0}, {2: 1.0}, {3: 1.0, 1: 0.2}]

# Fake query embeddings: "izin" means the same as article 0; "güvence" shares a word with article 2 only.
QUERIES = {
    "izin": (np.array([[1.0, 0.1, 0.0]]), [{9: 1.0}]),
    "güvence": (np.array([[0.0, 0.0, 0.1]]), [{3: 1.0}]),
    "izin güvence": (np.array([[1.0, 0.0, 0.0]]), [{3: 1.0}]),
}


@pytest.fixture
def index(monkeypatch):
    monkeypatch.setattr(vectors, "embed", lambda texts: QUERIES[texts[0]])
    return vectors.VectorIndex(ARTICLES, DENSE, SPARSE)


def keys(hits):
    return [(a.law_number, a.article_number) for a, _ in hits]


def test_dense_ranks_by_meaning(index):
    assert keys(index.search("izin", 2, "dense")) == [("4857", "53"), ("6698", "11")]


def test_sparse_matches_shared_words_only(index):
    assert keys(index.search("güvence", 3, "sparse")) == [("6098", "342")]


def test_hybrid_fuses_both_lists(index):
    top2 = set(keys(index.search("izin güvence", 2, "hybrid")))
    assert top2 == {("4857", "53"), ("6098", "342")}


def test_law_filter(index):
    assert keys(index.search("izin", 3, "dense", law_number="6698")) == [("6698", "11")]


def test_cache_must_match_articles():
    with pytest.raises(AssertionError):
        vectors.VectorIndex(ARTICLES, DENSE[:2], SPARSE)


def test_article_text_carries_context():
    text = vectors.article_text(find_article("6098", "313"))
    assert text.startswith("Türk Borçlar Kanunu, madde 313: 1. Genel olarak")
    assert "Kira Sözleşmesi" in text


def test_search_reranked_reorders_candidates(index, monkeypatch):
    # The reranker prefers article 2 even though dense search ranks it last.
    monkeypatch.setattr(vectors, "rerank_scores", lambda q, arts: [0.9 if a.article_number == "342" else 0.1 for a in arts])
    hits = index.search_reranked("izin", 2, "dense")
    assert keys(hits)[0] == ("6098", "342")
    assert len(hits) == 2
