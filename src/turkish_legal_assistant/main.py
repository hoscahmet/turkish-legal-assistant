from fastapi import FastAPI, HTTPException
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelError,
    ModelRateLimitError,
    ModelRefusalError,
)

from . import agent, local
from .data import LAWS, find_article
from .models import Article, AskRequest, AskResponse, Law

app = FastAPI(title="Turkish Legal Assistant")
REFUSAL_ANSWER = "Bu soruya cevap veremiyorum."


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
    # LangChain maps every provider's errors to the same Model*Error classes.
    try:
        return agent.answer(req.question, req.top_k)
    except ModelRefusalError:
        return AskResponse(question=req.question, answer=REFUSAL_ANSWER, sources=[])
    except (agent.LLMNotConfigured, ModelAuthenticationError) as e:
        raise HTTPException(status_code=503, detail=f"LLM is not configured: {e}")
    except local.LocalModelUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except local.LocalModelError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ModelRateLimitError:
        raise HTTPException(status_code=503, detail="LLM rate limit reached, try again later")
    except ModelAPIError as e:
        # Gemini's free tier answers 503 "high demand" when overloaded: temporary, so 503 too.
        status = 503 if getattr(e, "code", None) == 503 else 502
        raise HTTPException(status_code=status, detail=f"LLM request failed: {e}")
    except (ModelError, agent.LLMError) as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")
