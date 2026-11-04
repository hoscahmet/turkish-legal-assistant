import anthropic
from fastapi import FastAPI, HTTPException

from . import llm
from .data import LAWS, find_article
from .models import Article, AskRequest, AskResponse, Law

app = FastAPI(title="Turkish Legal Assistant")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/laws")
def list_laws() -> list[Law]:
    return LAWS


# ":path" lets numbers like "47/A" contain a slash.
@app.get("/laws/{law_number}/articles/{article_number:path}")
def get_article(law_number: str, article_number: str) -> Article:
    article = find_article(law_number, article_number)
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")
    return article


@app.post("/ask")
def ask(req: AskRequest) -> AskResponse:
    try:
        return llm.answer(req.question, req.top_k)
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=503, detail="LLM is not configured: set ANTHROPIC_API_KEY")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=503, detail="LLM rate limit reached, try again later")
    except (anthropic.APIError, llm.LLMError) as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")
