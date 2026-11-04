from pydantic import BaseModel, Field


class Article(BaseModel):
    # Strings, not ints: some articles are numbered like "Geçici 1" or "Ek 2".
    law_number: str
    law_name: str
    article_number: str
    title: str
    text: str
    section: str = ""  # e.g. "DÖRDÜNCÜ BÖLÜM: Kira Sözleşmesi > BİRİNCİ AYIRIM: Genel Hükümler"
    source_url: str


class Law(BaseModel):
    number: str
    name: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


class SearchResult(BaseModel):
    article: Article
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Article]


class Citation(BaseModel):
    law_number: str
    article_number: str


class LegalAnswer(BaseModel):
    """The JSON shape Claude must return (structured output)."""

    answer: str = Field(description="Answer to the user's question, in Turkish.")
    citations: list[Citation] = Field(description="Articles the answer is based on.")
