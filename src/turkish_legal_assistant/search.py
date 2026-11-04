import re

from .data import ARTICLES
from .models import SearchResult
from .text import tr_lower


def tokenize(text: str) -> set[str]:
    # Drop 1-2 letter words ("ve", "bu") so they don't dominate the overlap score.
    return {w for w in re.findall(r"\w+", tr_lower(text)) if len(w) > 2}


def search(query: str, top_k: int, law_number: str | None = None) -> list[SearchResult]:
    # shortcut: naive keyword overlap, replaced by hybrid search + reranker in Phase 3.
    q = tokenize(query)
    if not q:
        return []
    scored = [
        SearchResult(article=a, score=len(q & tokenize(f"{a.title} {a.text}")) / len(q))
        for a in ARTICLES
        if law_number is None or a.law_number == law_number
    ]
    hits = sorted((r for r in scored if r.score > 0), key=lambda r: r.score, reverse=True)
    return hits[:top_k]
