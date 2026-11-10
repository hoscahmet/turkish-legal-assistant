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
- Faz 6: Küçük modeli QLoRA ile fine-tune, vLLM ile servis (tamamlandı 2026-11-10, main'e birleştirildi)
  - Donanım: Windows PC, RTX 4070 Ti 12 GB (~11 GB boş), WSL2 Ubuntu, RAM 11 GB (.wslconfig), CUDA 13.3
  - Öğretmen: Qwen/Qwen3-8B-AWQ (vLLM), öğrenci: Qwen/Qwen3-1.7B (QLoRA). İkisi de Apache 2.0;
    Claude/Gemini çıktısıyla eğitim yok (kullanım şartları)
  - Öğrenci kanunu değil, RAG cevap adımını öğrenir: soru + bulunan maddeler -> LegalAnswer JSON
  - Yapıldı: veri üretimi (data/finetune/: train 460, eval 53; eval kaynak maddeleri train'de yok).
  - Yapıldı: QLoRA eğitimi (finetune.py, 3 epoch, 23 dk, adaptör models/ altında, gitignored).
    Eval loss 0.499 / 0.479 / 0.493: 3. epoch hafif overfit, gerekirse 2 epoch'a in.
  - Yapıldı: karşılaştırma (compare.py, eval split, 53 soru; hakemsiz ölçütler):
    temel 1.7B valid %21 / grounded %0; öğrenci valid %100, grounded %96, source %69, agree 0.64, 0.50 sn/soru;
    öğretmen valid %100, grounded %98, source %82, agree 0.87 (referansı kendisi yazdı), 2.38 sn/soru.
    gemini-3.5-flash-lite valid %100, grounded %100, source %77, agree 0.68 (6.24 sn/soru, 5 sn'si rate-limit beklemesi).
    Öğrencide Türkçe ek hataları ("madde 5'ye") ve tekrar var.
  - Yapıldı: LLM_PROVIDER=local (local.py) ve serve.py (vLLM OpenAI sunucusu, LoRA adı "legal", port 8001).
    Öğrenci tool calling bilmez: local modda agent yok, retrieve -> eğitim prompt'u -> JSON şeması zorunlu.
    4B bf16 servis için KV cache'e yer bırakmaz: --quantization bitsandbytes (eğitimdeki NF4 ile aynı).
    API'yi EMBED_DEVICE=cpu ile çalıştır, BGE-M3 vLLM ile GPU'yu paylaşmasın.
  - Yapıldı: Qwen3-4B öğrenci (52 dk, eval loss 0.385 / 0.387 / 0.405; 1. epoch en iyi, 3. hafif overfit).
    compare (4B 4-bit yüklenir): base-4b valid %100, grounded %62, source %49, agree 0.40;
    student-4b valid %100, grounded %100, source %77, agree 0.70, 0.92 sn/soru (Gemini ile başa baş).
    Uçtan uca /ask çalışıyor (local, 4B): ilk istek ~20 sn (BGE-M3 CPU'ya yükleniyor), sonra ~2 sn.
    Hata örneği: kira artışı sorusunda TBK 344'teki TÜFE sınırını "belirtilmemiş" dedi (içerik doğruluğu Faz 7).
  - WSL: vLLM sunucusu V2 model runner seçiyor, pinned memory (UVA) yok -> VLLM_USE_V2_MODEL_RUNNER=0 (serve.py)
  - HF indirmesi xet ile takıldı; HF_HUB_DISABLE_XET=1 ile klasik HTTP çalıştı
  - BGE-M3 ve vLLM aynı anda GPU'ya sığmaz: veri üretimi 3 ayrı process (questions/retrieve/answers)
  - WSL'de CUDA toolkit (nvcc) yok: FlashInfer sampler kapalı (VLLM_USE_FLASHINFER_SAMPLER=0)
  - Öğretmen cevaplarda thinking açık; kapalıyken maddelerle çelişen cevaplar yazıyordu
  - 1 epoch denemesi (2026-11-10): `finetune --size 4b --epochs 1`, eval loss 0.390, agree 0.67 (3 epoch 0.70).
    3 epoch'luk adaptör varsayılan kaldı; loss farkı cevap kalitesine yansımadı.
- Faz 7: LLM-as-a-judge ve CI kalite kapısı (tamamlandı 2026-11-10, dal: phase-7/judge); RAGAS ve Langfuse yapılmadı
  - judge.py: hakem cevabı iddialara ayırır (supported/contradicted/unsupported) + complete. Kararlar
    data/finetune/judge/<hakem>/<model>.jsonl (commit'li); aynı çıktı + aynı hakem ise yeniden kullanılır.
  - İki hakem: gemini-3.5-flash (ana hakem) ve yerel Qwen3-8B-AWQ (ücretsiz). Gemini 3.5 Flash: student-4b doğru %42,
    çelişkili %34; Flash-Lite %83 / %0; teacher %58 / %28. Qwen hakemi student-4b'ye %68 verdi (fazla hoşgörülü).
  - Hakemler ~1/3 cevapta ayrışıyor; 19 vaka elle incelendi, ~16'sında Gemini haklı. Gemini kendi ailesini hafif kayırıyor.
    Öğrencinin hataları: maddeyi yanlış duruma uygulama (askerlik maddesi, kıdem şartı), süre/koşul atlama.
  - Gemini hakemi student, base-4b, base için çalışmadı (fatura sorunu); student-4b-1ep 52/53.
  - ÜCRET: Gemini 3.5 Flash ile ~211 istek kullanıcıya 450 TL'ye mal oldu; ücretli API'yi fiyatı ölçüp TL olarak
    gösterip onay almadan asla çalıştırma. Varsayılan hakem local.
  - CI: .github/workflows/ci.yml (uv sync --no-group finetune, pytest). tests/test_quality.py: student-4b atıf ve
    Gemini hakem puanları eşiklerin altına düşerse ya da çıktılar yeniden puanlanmadan değişirse başarısız.
  - Push yapılamadı (WSL'de git kimlik bilgisi yok); main origin'in önünde.
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
- QLoRA training (GPU): `uv run python -m turkish_legal_assistant.finetune --size 1.7b|4b` (1.7B ~25 min, writes `models/qwen3-<size>-legal-lora/`)
- Serve the student (GPU): `uv run python -m turkish_legal_assistant.serve --size 4b`, then `LLM_PROVIDER=local EMBED_DEVICE=cpu uv run uvicorn turkish_legal_assistant.main:app` (`LOCAL_LLM_URL` overrides `http://localhost:8001/v1`)
- Compare models (GPU, one per process): `uv run python -m turkish_legal_assistant.compare base|student|base-4b|student-4b|teacher|gemini`, then `... compare report` (outputs in `data/finetune/compare/`)
- Judge answers (LLM-as-a-judge): `uv run python -m turkish_legal_assistant.judge [model ...] --judge local|gemini`, then `... judge report --judge gemini` (local = Qwen3-8B on the GPU, free; gemini needs a paid key)
- Test: `uv run pytest`
