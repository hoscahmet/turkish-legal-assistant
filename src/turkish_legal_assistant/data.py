from pydantic import TypeAdapter

from .ingest import DATA_DIR
from .models import Article, Law

# Built by `uv run python -m turkish_legal_assistant.ingest` from mevzuat.gov.tr.
ARTICLES: list[Article] = TypeAdapter(list[Article]).validate_json(
    (DATA_DIR / "articles.json").read_bytes()
)

LAWS: list[Law] = list({a.law_number: Law(number=a.law_number, name=a.law_name) for a in ARTICLES}.values())


def find_article(law_number: str, article_number: str) -> Article | None:
    return next(
        (a for a in ARTICLES if a.law_number == law_number and a.article_number == article_number),
        None,
    )
