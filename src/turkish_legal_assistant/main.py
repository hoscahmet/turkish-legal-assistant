import re

from fastapi import FastAPI, HTTPException

from .data import ARTICLES, LAWS
from .models import Article, AskRequest, AskResponse, Law, SearchResult
from .text import tr_lower

app = FastAPI(title="Turkish Legal Assistant")


def tokenize(text: str) -> set[str]:
    # Drop 1-2 letter words ("ve", "bu") so they don't dominate the overlap score.
    return {w for w in re.findall(r"\w+", tr_lower(text)) if len(w) > 2}


def search(question: str, top_k: int) -> list[SearchResult]:
    # shortcut: naive keyword overlap, replaced by hybrid search + reranker in Phase 3.
    q = tokenize(question)
    scored = [
        SearchResult(article=a, score=len(q & tokenize(f"{a.title} {a.text}")) / len(q))
        for a in ARTICLES
    ] if q else []
    hits = sorted((r for r in scored if r.score > 0), key=lambda r: r.score, reverse=True)
    return hits[:top_k]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/laws")
def list_laws() -> list[Law]:
    return LAWS


@app.get("/laws/{law_number}/articles/{article_number}")
def get_article(law_number: str, article_number: str) -> Article:
    for a in ARTICLES:
        if a.law_number == law_number and a.article_number == article_number:
            return a
    raise HTTPException(status_code=404, detail="Article not found")


@app.post("/ask")
def ask(req: AskRequest) -> AskResponse:
    sources = search(req.question, req.top_k)
    if sources:
        refs = ", ".join(f"{s.article.law_name} madde {s.article.article_number}" for s in sources)
        answer = f"Sorunuzla ilgili maddeler: {refs}."
    else:
        answer = "Sorunuzla ilgili bir madde bulunamadı."
    return AskResponse(question=req.question, answer=answer, sources=sources)
