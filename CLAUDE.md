# turkish-legal-assistant

Question answering system over Turkish legislation.

## Kurallar
- Kullanıcıyla sohbet (açıklamalar, ara notlar) Türkçe. Kullanıcı öğreniyor: her değişikliği kısaca açıkla.
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
- Faz 4: LangGraph agent (tamamlandı)
- Faz 5: MCP server (tamamlandı)
- Faz 6: Küçük modeli QLoRA ile fine-tune, vLLM ile servis (devam ediyor, dal: phase-6/finetune)
  - Donanım: Windows PC, RTX 4070 Ti 12 GB (~11 GB boş), WSL2 Ubuntu, RAM 11 GB (.wslconfig), CUDA 13.3
  - Öğretmen: Qwen/Qwen3-8B-AWQ (vLLM), öğrenci: Qwen/Qwen3-1.7B (QLoRA). İkisi de Apache 2.0;
    Claude/Gemini çıktısıyla eğitim yok (kullanım şartları)
  - Öğrenci kanunu değil, RAG cevap adımını öğrenir: soru + bulunan maddeler -> LegalAnswer JSON
  - Yapıldı: veri üretimi (data/finetune/: train 460, eval 53; eval kaynak maddeleri train'de yok).
  - Yapıldı: QLoRA eğitimi (finetune.py, 3 epoch, 23 dk, adaptör models/ altında, gitignored).
    Eval loss 0.499 / 0.479 / 0.493: 3. epoch hafif overfit, gerekirse 2 epoch'a in.
    Sırada: öğrenci/öğretmen/Gemini karşılaştırması (eval split), vLLM servis ve LLM_PROVIDER=local
  - BGE-M3 ve vLLM aynı anda GPU'ya sığmaz: veri üretimi 3 ayrı process (questions/retrieve/answers)
  - WSL'de CUDA toolkit (nvcc) yok: FlashInfer sampler kapalı (VLLM_USE_FLASHINFER_SAMPLER=0)
  - Öğretmen cevaplarda thinking açık; kapalıyken maddelerle çelişen cevaplar yazıyordu
- Faz 7: RAGAS, LLM-as-a-judge, Langfuse, CI'da regression testi
- Faz 8: Docker Compose, demo arayüz, README

## Commands
- Install: `uv sync`
- Ingest laws: `uv run python -m turkish_legal_assistant.ingest` (writes `data/articles.json`; delete `data/raw/` to re-download)
- LLM: `/ask` runs the LangGraph agent in `agent.py`. `LLM_PROVIDER=gemini` (default, free tier, needs `GEMINI_API_KEY`; `GEMINI_MODEL` overrides the model) or `claude` (needs `ANTHROPIC_API_KEY`). Tests use a scripted fake model, never a real API.
- Embed articles: `uv run python -m turkish_legal_assistant.vectors` (BGE-M3, writes `data/embeddings_*`, gitignored). Tests never load the model (see `tests/conftest.py`).
- Evaluate retrieval: `uv run python -m turkish_legal_assistant.evaluate` (golden set: `data/golden.json`, Turkish questions with expected articles)
- MCP server (stdio): `uv run python -m turkish_legal_assistant.mcp_server` (mcp 2.x: `MCPServer`, not `FastMCP`; tests use the in-process `Client(server)`)
- Run: `uv run uvicorn turkish_legal_assistant.main:app --reload`
- Fine-tuning data (GPU): `uv run python -m turkish_legal_assistant.finetune_data questions|retrieve|answers` (run in order; `--limit N` for a trial; ~25 min total)
- QLoRA training (GPU): `uv run python -m turkish_legal_assistant.finetune` (~25 min, writes `models/qwen3-1.7b-legal-lora/`)
- Test: `uv run pytest`
