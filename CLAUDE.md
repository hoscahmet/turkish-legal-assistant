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
- Faz 7: LLM-as-a-judge ve CI kalite kapısı (tamamlandı 2026-11-11, main'e birleştirildi); RAGAS ve Langfuse yapılmadı
  - judge.py: hakem cevabı iddialara ayırır (supported/contradicted/unsupported) + complete. Kararlar
    data/finetune/judge/<hakem>/<model>.jsonl (commit'li); aynı çıktı + aynı hakem ise yeniden kullanılır.
  - İki hakem: gemini-3.5-flash (ana hakem) ve yerel Qwen3-8B-AWQ (ücretsiz). Gemini 3.5 Flash: student-4b doğru %42,
    çelişkili %34; Flash-Lite %83 / %0; teacher %58 / %28. Qwen hakemi student-4b'ye %68 verdi (fazla hoşgörülü).
  - Hakemler ~1/3 cevapta ayrışıyor; 19 vaka elle incelendi, ~16'sında Gemini haklı. Gemini kendi ailesini hafif kayırıyor.
    Öğrencinin hataları: maddeyi yanlış duruma uygulama (askerlik maddesi, kıdem şartı), süre/koşul atlama.
  - Gemini hakemi 2026-11-11'de student (doğru %34, çelişkili %43) ve student-4b-1ep (53/53, %34 / %38) için tamamlandı.
    base-4b 32/53'te bırakıldı, base hiç yok: hesap yine "outstanding balance" verdi, kullanıcı bırakmayı seçti.
  - ÜCRET: Gemini 3.5 Flash ile ~211 istek kullanıcıya 450 TL'ye mal oldu; ücretli API'yi fiyatı ölçüp TL olarak
    gösterip onay almadan asla çalıştırma. Varsayılan hakem local.
  - CI: .github/workflows/ci.yml (uv sync --no-group finetune, pytest). tests/test_quality.py: student-4b atıf ve
    Gemini hakem puanları eşiklerin altına düşerse ya da çıktılar yeniden puanlanmadan değişirse başarısız.
  - Push yapılamadı (WSL'de git kimlik bilgisi yok); main origin'in önünde.
- Faz 8: Docker Compose, demo arayüz, README (main'e merge edildi 2026-11-11)
  - Demo arayüz: src/turkish_legal_assistant/static/index.html (düz HTML+JS, derleme yok), FastAPI `/` adresinden sunar.
    `/info` hangi modelin cevap verdiğini döner (agent.model_label). Yeni bağımlılık yok (Streamlit/Gradio bilinçli olarak eklenmedi).
  - Uçtan uca denendi (2026-11-11, local 4B): sayfa + /ask çalışıyor, ~2-3 sn/soru. vLLM kapalıyken /ask 503 + açıklama.
  - Docker: Dockerfile (api, --no-group finetune, EMBED_DEVICE=cpu), docker/vllm.Dockerfile (vllm-openai v0.22.1 + bitsandbytes),
    compose.yaml (vllm servisi `--profile local` ile açılır; data/, models/, HF önbelleği mount edilir). .env.example.
    tests/test_docker.py: compose'daki vLLM bayrakları serve.py'deki serve_command ile aynı kalmalı.
  - Docker testi (2026-11-11, Docker Desktop WSL entegrasyonu, `--profile local`): iki imaj build oldu, vLLM GPU'yu gördü ve
    healthy oldu, API ardından açıldı; /, /info, /health, /laws, /ask çalışıyor (ilk soru ~21 sn, sonrakiler ~3 sn).
    vllm imajı 35.2 GB. Konteynerler açıkken GPU ~10.5/12.3 GB dolu.
    Profilsiz Gemini yolu (`docker compose up`) Docker'da denenmedi (ücretli API).
  - API imajı 17.5 GB -> 2.62 GB (2026-11-11): Dockerfile uv.lock'u export eder, torch'un CUDA tekerleklerini (nvidia-*, cuda-*, triton)
    ayıklar, torch'u aynı sürümle PyTorch CPU index'inden kurar; uv önbelleği cache mount'ta kalır (eski katmanın ~5.8 GB'ı önbellekti).
    Proje editable kurulmalı (`-e .`): kod data/ yolunu kendi konumuna göre buluyor, yoksa /ask ve /laws FileNotFoundError verir.
    tests/test_docker.py: ayıklama deseni kilitteki tüm CUDA paketlerini kapsamalı.
  - WSL interop bozulursa (`UtilAcceptVsock ... accept4 failed 110`) docker build "error getting credentials" verir (kimlik yardımcısı
    bir Windows exe'si). Kalıcı çözüm `wsl --shutdown`; geçici: boş config.json içeren bir DOCKER_CONFIG dizini (imajlar herkese açık).
- Faz 9: öğrencinin cevap kalitesi (tamamlandı 2026-11-11, dal phase-9/student-v2)
  - Ölçüm: 53 eval sorusu, yerel hakem, her yapılandırma 3 çalıştırma (tek çalıştırma ±5 puan oynar).
    İlk adaptör %65 doğru; sıcaklık 0 / tekrar cezası / 3 madde fark yaratmadı; ilk adaptör + düşünme %64 (JSON'un %11'i bozuk);
    v2 adaptör düşünmeden %65; v2 adaptör + düşünme %80 (72/83/85). Commit'li resmi çalıştırma %72 (4 çalıştırma ort. %78).
  - verify.py: öğretmenin 460 eğitim cevabını yerel hakem puanladı, 108'i kusurlu (70 eksik, 25 çelişkili, 13 dayanaksız).
    Kusurlulara 3 aday yazdırılır, hakemden geçen ilki alınır (91/113 onarıldı); eğitim maddelerine 683 yeni soru;
    yalnızca tam + tüm iddiaları doğrulanan + köşeli kod/tekrar içermeyen cevaplar girer -> data/finetune/v2/train.jsonl (815).
    Ara dosyalar data/finetune/v2/work/ (gitignored). Eval split ve soruları aynı.
  - Sunulan model: models/qwen3-4b-v2-legal-lora (finetune --size 4b --data 2, 89 dk, eval loss 0.376/0.376/0.401),
    local.py düşünme açık (LOCAL_THINKING=0 ile kapanır; ilk adaptörler için), vLLM `--reasoning-parser qwen3` (serve.py + compose.yaml).
    ~1.8 sn/soru. Cevap Çinceye kayarsa local.py bir kez yeniden sorar.
    Docker'da yeniden test edildi (2026-11-11): ilk soru ~30 sn, sonrakiler 5-10 sn (tek istek; 1.8 sn toplu çalıştırma hızı).
    Demo sorusu "ev sahibi oğlu oturacak diye çıkarabilir mi" top_k=3 ile TBK 350'yi bulamıyor, model "maddeler cevaplamıyor" diyor (arama sorunu).
  - Kalite kapısı artık student-4b-v2'ye bakıyor: atıf + yerel hakem zorunlu, Gemini testi kararlar gelene kadar skip.
    Gemini ile doğrulanmadı (ücretli ya da günde 20 istek); kullanıcıdan onay almadan çalıştırma.
  - Deney betikleri ve ham çıktılar models/experiments-2026-11-11/ altında (gitignored).
  - WSL: çevrimdışı vLLM (LLM(...)) LoRA ile V2 runner seçiyor -> vllm_generate VLLM_USE_V2_MODEL_RUNNER=0 ayarlar;
    kendi betiğinde `if __name__ == "__main__":` şart (vLLM alt süreç başlatır).

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
- Verified training set (GPU, one stage per process): `uv run python -m turkish_legal_assistant.verify questions|retrieve|answers|judge|repair|judge|build` (writes `data/finetune/v2/train.jsonl`), then `... finetune --size 4b --data 2`
- Serve the student (GPU): `uv run python -m turkish_legal_assistant.serve --size 4b` (the `--data 2` adapter, thinking on), then `LLM_PROVIDER=local EMBED_DEVICE=cpu uv run uvicorn turkish_legal_assistant.main:app` (`LOCAL_LLM_URL` overrides `http://localhost:8001/v1`)
- Compare models (GPU, one per process): `uv run python -m turkish_legal_assistant.compare base|student|base-4b|student-4b|student-4b-v2|teacher|gemini`, then `... compare report` (outputs in `data/finetune/compare/`)
- Judge answers (LLM-as-a-judge): `uv run python -m turkish_legal_assistant.judge [model ...] --judge local|gemini`, then `... judge report --judge gemini` (local = Qwen3-8B on the GPU, free; gemini needs a paid key)
- Test: `uv run pytest`
