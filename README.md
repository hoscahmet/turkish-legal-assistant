# turkish-legal-assistant

A question answering system over Turkish legislation, built step by step as an AI engineering portfolio project.

Target laws: 4857 Labour Law, 6698 Personal Data Protection Law (KVKK), 6502 Consumer Protection Law, 6098 Turkish Code of Obligations (lease section).

## Status

- Phase 0: FastAPI skeleton with Pydantic models and naive keyword search.
- Phase 1: `ingest.py` downloads the four laws from mevzuat.gov.tr, splits them into one chunk per article (348 articles), and stores title, section path and source URL as metadata in `data/articles.json`.

See `CLAUDE.md` for the full roadmap (scraping, LLM, hybrid search, agents, MCP, fine-tuning, evaluation, deployment).

## Quick start

```bash
uv sync
uv run python -m turkish_legal_assistant.ingest  # optional: rebuild data/articles.json
uv run uvicorn turkish_legal_assistant.main:app --reload
uv run pytest
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/laws` | List laws |
| GET | `/laws/{law_number}/articles/{article_number}` | Get one article |
| POST | `/ask` | Ask a question, e.g. `{"question": "Yıllık izin kaç gündür?", "top_k": 3}` |

Answers and law texts are in Turkish.
