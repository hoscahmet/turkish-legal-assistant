import pytest

from turkish_legal_assistant.data import find_article
from turkish_legal_assistant.evaluate import evaluate, load_golden, recall_at_k, reciprocal_rank

A, B, C = ("4857", "1"), ("4857", "2"), ("4857", "3")


def test_recall_at_k():
    assert recall_at_k([A, B, C], {B, C}, k=1) == 0
    assert recall_at_k([A, B, C], {B, C}, k=2) == 0.5
    assert recall_at_k([A, B, C], {B, C}, k=3) == 1


def test_reciprocal_rank():
    assert reciprocal_rank([A, B, C], {B}) == 0.5
    assert reciprocal_rank([A, B, C], {("6698", "1")}) == 0


def test_evaluate_with_perfect_and_empty_retrievers():
    golden = [{"question": "q", "relevant": [list(A)]}]
    assert evaluate(lambda q, k: [A], golden)["mrr"] == 1
    empty = evaluate(lambda q, k: [], golden)
    assert empty["recall@10"] == 0
    assert empty["misses"] == ["q"]


@pytest.mark.parametrize("item", load_golden(), ids=lambda item: item["question"][:40])
def test_golden_articles_exist(item):
    for law_number, article_number in item["relevant"]:
        assert find_article(law_number, article_number) is not None
