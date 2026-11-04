"""Measure retrieval quality on the golden dataset with Recall@k and MRR.

Run: uv run python -m turkish_legal_assistant.evaluate
"""

import json
from collections.abc import Callable
from statistics import mean

from .ingest import DATA_DIR
from .search import search

ArticleKey = tuple[str, str]  # (law_number, article_number)
# A retriever takes a query and k, and returns article keys ranked best first.
Retriever = Callable[[str, int], list[ArticleKey]]
K_VALUES = (1, 3, 5, 10)


def recall_at_k(ranked: list[ArticleKey], relevant: set[ArticleKey], k: int) -> float:
    """Share of the relevant articles that appear in the top k results."""
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[ArticleKey], relevant: set[ArticleKey]) -> float:
    """1 / position of the first relevant result; 0 if none was found."""
    return next((1 / i for i, key in enumerate(ranked, 1) if key in relevant), 0.0)


def load_golden() -> list[dict]:
    return json.loads((DATA_DIR / "golden.json").read_text(encoding="utf-8"))


def evaluate(retrieve: Retriever, golden: list[dict]) -> dict:
    rows = []
    for item in golden:
        relevant = {tuple(r) for r in item["relevant"]}
        ranked = retrieve(item["question"], max(K_VALUES))
        rows.append((item["question"], relevant, ranked))
    scores = {f"recall@{k}": mean(recall_at_k(r, rel, k) for _, rel, r in rows) for k in K_VALUES}
    scores["mrr"] = mean(reciprocal_rank(r, rel) for _, rel, r in rows)
    scores["misses"] = [q for q, rel, r in rows if not set(r[:5]) & rel]  # nothing relevant in top 5
    return scores


def keyword_retriever(query: str, k: int) -> list[ArticleKey]:
    return [(r.article.law_number, r.article.article_number) for r in search(query, k)]


RETRIEVERS: dict[str, Retriever] = {"keyword": keyword_retriever}


def main() -> None:
    golden = load_golden()
    print(f"{len(golden)} questions\n")
    for name, retrieve in RETRIEVERS.items():
        scores = evaluate(retrieve, golden)
        print(f"{name:10}" + "  ".join(f"{m}={scores[m]:.2f}" for m in [*(f"recall@{k}" for k in K_VALUES), "mrr"]))
        print(f"  misses (nothing relevant in top 5): {len(scores['misses'])}")
        for q in scores["misses"]:
            print(f"   - {q}")


if __name__ == "__main__":
    main()
