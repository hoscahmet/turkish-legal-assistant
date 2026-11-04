# turkish-legal-assistant

Question answering system over Turkish legislation.

## Kurallar
- Kod, isimler, yorumlar, testler, README ve commit mesajları İngilizce.
- Kanun metinleri, test soruları ve son kullanıcıya verilen cevaplar Türkçe.
- Paket yönetimi uv ile. Her yeni özellik için pytest testi yazılır.
- Türkçe metinde küçük harfe çevirmek için str.lower() değil,
  tr_lower() kullanılır (İ ve I harfleri için).

## Yol haritası
- Faz 0: FastAPI iskeleti, Pydantic modelleri (tamamlandı)
- Faz 1: mevzuat.gov.tr'den kanun metinlerini çekmek, madde bazlı
  chunking ve metadata. Kanunlar: 4857 İş Kanunu, 6698 KVKK,
  6502 Tüketicinin Korunması Hakkında Kanun, 6098 TBK (kira bölümü) (tamamlandı)
- Faz 2: LLM API, structured output, tool calling (tamamlandı)
- Faz 3: Qdrant, BGE-M3 embedding, hybrid search, reranker,
  golden dataset ile Recall@k / MRR ölçümü (tamamlandı; en iyi: dense, MRR 0.77)
- Faz 4: LangGraph agent
- Faz 5: MCP server
- Faz 6: Küçük modeli QLoRA ile fine-tune, vLLM ile servis
- Faz 7: RAGAS, LLM-as-a-judge, Langfuse, CI'da regression testi
- Faz 8: Docker Compose, demo arayüz, README

## Commands
- Install: `uv sync`
- Ingest laws: `uv run python -m turkish_legal_assistant.ingest` (writes `data/articles.json`; delete `data/raw/` to re-download)
- LLM: `/ask` runs a tool loop in `llm.py`. `LLM_PROVIDER=gemini` (default, free tier, needs `GEMINI_API_KEY`) or `claude` (needs `ANTHROPIC_API_KEY`). Tests use fake clients, never a real API.
- Embed articles: `uv run python -m turkish_legal_assistant.vectors` (BGE-M3, writes `data/embeddings_*`, gitignored). Tests never load the model (see `tests/conftest.py`).
- Evaluate retrieval: `uv run python -m turkish_legal_assistant.evaluate` (golden set: `data/golden.json`, Turkish questions with expected articles)
- Run: `uv run uvicorn turkish_legal_assistant.main:app --reload`
- Test: `uv run pytest`
