<div align="center">

# ⚖️ Türk Hukuk Asistanı

**Türk mevzuatı hakkında günlük Türkçeyle soru sorun; ilgili maddeyi gösteren kısa cevaplar alın.**

[![English](https://img.shields.io/badge/🇬🇧_English-read-1f6feb?style=for-the-badge)](README.md)
[![Türkçe](https://img.shields.io/badge/🇹🇷_Türkçe-seçili-e30a17?style=for-the-badge)](README.tr.md)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-agent-1C3C3C?logo=langchain&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-vector_search-DC244C)
![BGE-M3](https://img.shields.io/badge/BGE--M3-embeddings-FFD21E?logo=huggingface&logoColor=black)
![MCP](https://img.shields.io/badge/MCP-server-8A2BE2)
![vLLM](https://img.shields.io/badge/vLLM-QLoRA_student-30A2FF)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![uv](https://img.shields.io/badge/uv-package_manager-DE5FE9)
[![Lisans: MIT](https://img.shields.io/badge/Lisans-MIT-green)](LICENSE)

</div>

---

## 💬 Örnek

> **Soru:** *Ev sahibi oğlu oturacak diye beni evden çıkarabilir mi?*
>
> **Cevap:** Evet, ama ancak dava açarak ve oğlunun evi kullanma zorunluluğu varsa. Bu sebeple çıkarırsa, haklı bir sebep olmadıkça evi 3 yıl boyunca başkasına kiralayamaz.
>
> 📎 **Kaynaklar:** Türk Borçlar Kanunu, md. 350 · md. 355

Her cevap modelin hafızasından değil, kanun metninin kendisinden gelir. Var olmayan maddelere yapılan atıflar otomatik olarak silinir.

## ✨ Neler yapıyor

| | |
|---|---|
| 📚 **4 kanun, 348 madde** | İş Kanunu (4857), KVKK (6698), Tüketicinin Korunması Hakkında Kanun (6502), Türk Borçlar Kanunu kira hükümleri (6098) |
| 🔎 **Anlamsal arama** | "Patron maaş vermiyor" diye sorduğunuzda "işveren ücreti ödemiyor" diyen maddeyi bulur |
| 🤖 **Agent** | Model arar, okur, gerekirse tekrar arar, sonra cevaplar |
| 🔌 **MCP sunucusu** | Aramayı Claude Desktop, Claude Code ya da herhangi bir MCP istemcisine bağlayın |
| 🛡️ **Halüsinasyon koruması** | Kaynak olarak sadece gerçekten var olan maddeler gösterilir |
| 🔄 **Model bağımsız** | Gemini (ücretsiz katman), Claude ya da yerelde eğitilmiş Qwen3-4B, tek ayarla değişir |
| 🖥️ **Demo sayfası** | Tarayıcıdan sorun, dayanılan maddeleri okuyun; sayfa hangi modelin cevap verdiğini gösterir |
| 🐳 **Docker Compose** | Tek komut API'yi ve demo sayfasını açar; GPU kullanan ikinci bir konteyner yerel modeli sunar |

## 🏗️ Nasıl çalışıyor

```mermaid
flowchart LR
    Q(["❓ Soru"]) --> A{"🤖 Agent"}
    A -- arama --> S["🔎 Anlamsal arama<br/>BGE-M3 + Qdrant"]
    S --> D[("📚 348 madde")]
    D --> A
    A -- bilgi yeterli --> R["✍ Cevap<br/>JSON + atıflar"]
    R --> G["🛡 Atıf kontrolü"]
    G --> OUT(["✅ Cevap + kaynaklar"])

    classDef blue fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a
    classDef green fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef yellow fill:#fef9c3,stroke:#a16207,color:#713f12
    classDef purple fill:#f3e8ff,stroke:#7e22ce,color:#581c87
    class Q,OUT green
    class A purple
    class S,D blue
    class R,G yellow
```

Bu yaklaşımın adı **RAG** (Retrieval-Augmented Generation): önce doğru maddeler *bulunur*, sonra model cevabı *sadece bu maddelere dayanarak* yazar.

## 🗺️ Yol haritası

| Faz | Konu | Durum |
|:---:|---|:---:|
| 0 | FastAPI iskeleti, Pydantic modelleri | ✅ |
| 1 | mevzuat.gov.tr'den kanunları çekmek, madde madde bölmek | ✅ |
| 2 | Tool calling ve structured output ile LLM | ✅ |
| 3 | BGE-M3 embedding, Qdrant, hybrid arama, reranker, ölçüm | ✅ |
| 4 | LangGraph agent | ✅ |
| 5 | MCP sunucusu | ✅ |
| 6 | Küçük bir modeli QLoRA ile eğitmek, vLLM ile sunmak | ✅ |
| 7 | LLM-as-a-judge, CI kalite kapısı | ✅ |
| 7+ | RAGAS, Langfuse izleme | ⏳ |
| 8 | Docker Compose, demo arayüzü | ✅ |

## 📊 Arama sonuçları

Doğru maddesi bilinen 39 günlük Türkçe soru üzerinde ölçüldü.

| Yöntem | Recall@1 | Recall@5 | MRR |
|---|:---:|:---:|:---:|
| Anahtar kelime (başlangıç) | 0.19 | 0.38 | 0.30 |
| BGE-M3 sparse | 0.33 | 0.45 | 0.42 |
| BGE-M3 hybrid (RRF) | 0.38 | **0.87** | 0.61 |
| 🏆 **BGE-M3 dense** (kullanılan) | **0.64** | 0.86 | **0.77** |
| Dense + bge-reranker-v2-m3 | 0.56 | 0.86 | 0.72 |

**Çıkarımlar**
- Anlamsal (dense) arama, anahtar kelime aramasını açık ara geçiyor (MRR 0.77'ye 0.30).
- Hybrid birleştirme ve reranker bu veride işe *yaramadı*. Reranker uzun ve genel maddeleri asıl cevabın önüne geçirme eğiliminde.
- 39 soruda 0.05'lik MRR farkı yaklaşık iki soruya denk geliyor; küçük farklar kesin sonuç sayılmaz.

## 🎓 Yerel model (fine-tuning)

Cevap adımı, tek bir oyuncu ekran kartında (RTX 4070 Ti, 12 GB) çalışan küçük, açık bir modelle de yapılabiliyor; hiçbir API gerekmez.

1. **Öğretmen verisi.** Qwen3-8B (AWQ, vLLM ile) üretilen soruları bulunan maddelere dayanarak cevaplar: 460 eğitim, 53 değerlendirme örneği. Değerlendirme sorularının dayandığı maddeler eğitimde hiç yer almaz. Veriyi yalnızca Apache-2.0 lisanslı modeller yazar; eğitimde Claude ya da Gemini çıktısı kullanılmaz.
2. **QLoRA.** Qwen3-4B 4 bit olarak yüklenip dondurulur. Küçük LoRA adaptörleri (ağırlıkların %0.8'i) *soru + maddeler*den cevap JSON'unu üretmeyi öğrenir. Kayıp yalnızca cevap üzerinden hesaplanır. Eğitim 52 dakika sürer.
3. **Sunum.** vLLM, temel modeli adaptörle birlikte OpenAI uyumlu bir sunucu olarak sunar. `LLM_PROVIDER=local` ile API maddeleri bulup tek adımda bu sunucuya gönderir, JSON şeması zorunlu tutulur.

53 değerlendirme sorusunda, her modele aynı prompt verilerek ölçüldü:

| Model | Geçerli JSON | Dayanaklı | Doğru kaynak | Uyum | sn/soru |
|---|:---:|:---:|:---:|:---:|:---:|
| Qwen3-1.7B, temel | %21 | %0 | %0 | 0.00 | 0.63 |
| Qwen3-1.7B, eğitilmiş | %100 | %96 | %69 | 0.64 | 0.50 |
| Qwen3-4B, temel | %100 | %62 | %49 | 0.40 | 0.94 |
| 🏆 **Qwen3-4B, eğitilmiş** (kullanılan) | %100 | **%100** | %77 | **0.70** | 0.92 |
| Qwen3-4B, 1 epoch eğitilmiş | %100 | %100 | %77 | 0.67 | 1.40 |
| Gemini 3.5 Flash-Lite | %100 | %100 | %77 | 0.68 | 6.24 |
| Qwen3-8B öğretmen | %100 | %98 | %82 | 0.87 | 2.38 |

*Dayanaklı:* her atıf verilen maddelerden biri. *Doğru kaynak:* soru hangi maddeden yazıldıysa ona atıf var (madde bulunmuşsa). *Uyum:* öğretmenin cevabına göre atıf F1'i (öğretmen bu yüzden doğal olarak yüksek çıkar).

**Çıkarımlar**
- Fine-tuning, olmayan maddelere atıf yapan 4B'lik bir modeli bu atıf ölçütlerinde Gemini ile başa baş hale getiriyor ve yerelde çalışıyor. İçerikte ise değil; bir sonraki bölüme bakın.
- 1 epoch daha düşük değerlendirme kaybı verdi (0.390'a karşı 0.405) ama daha iyi cevap vermedi; 3 epoch'luk adaptör kullanılıyor.
- Bu ölçütler atıfları kontrol eder, hukuki içeriği değil. Öğrenci yine de bir bilgiyi yanlış verebilir (örn. TBK 344'teki kira artışı TÜFE sınırını atladı); içerik değerlendirmesi bir sonraki bölümde.

## 🧑‍⚖️ Cevap kalitesi (LLM-as-a-judge)

Yukarıdaki atıf ölçütleri, doğru maddeye atıf yapan ama yanlış bilgi veren bir cevabı doğru cevaptan ayıramaz. Bu yüzden bir hakem model her cevabı verilen maddelerle birlikte okur, iddialara ayırır ve her iddiayı **doğrulanan**, **çelişen** (madde aksini söylüyor ya da cevap, maddelerde olan bir şeyin olmadığını söylüyor) veya **dayanaksız** diye işaretler. Ayrıca cevabın **tam** olup olmadığına bakar: soru için gereken kural, sayı ya da koşul verilmiş mi?

*Doğru:* tam ve hiç çelişen iddia yok. *Sadık:* bütün iddialar doğrulanıyor. *Çelişkili:* en az bir iddia maddelerle çelişiyor.

**Hakem: Gemini 3.5 Flash** (Qwen modellerinden bağımsız; puanlanan Flash-Lite'tan daha güçlü)

| Model | Doğru | Tam | Sadık | Çelişkili |
|---|:---:|:---:|:---:|:---:|
| Qwen3-1.7B, eğitilmiş | %34 | %43 | %47 | %43 |
| Qwen3-4B, eğitilmiş (kullanılan) | %42 | %47 | %58 | %34 |
| Qwen3-4B, 1 epoch eğitilmiş | %34 | %42 | %47 | %38 |
| Qwen3-8B öğretmen | %58 | %68 | %60 | %28 |
| Gemini 3.5 Flash-Lite | **%83** | **%83** | **%100** | **%0** |

Temel modeller Gemini ile puanlanmadı (maliyet yüzünden durduruldu); onlar için aşağıdaki yerel hakeme bakın.

**Hakem: Qwen3-8B** (yerel ve ücretsiz, ama öğrencinin eğitim verisini o yazdı)

| Model | Doğru | Tam | Sadık | Çelişkili |
|---|:---:|:---:|:---:|:---:|
| Qwen3-1.7B, temel | %9 | %9 | %19 | %2 |
| Qwen3-1.7B, eğitilmiş | %38 | %38 | %53 | %30 |
| Qwen3-4B, temel | %51 | %51 | %79 | %11 |
| Qwen3-4B, eğitilmiş (kullanılan) | %68 | %72 | %74 | %17 |
| Qwen3-4B, 1 epoch eğitilmiş | %58 | %60 | %81 | %9 |
| Qwen3-8B öğretmen | %81 | %81 | %81 | %8 |
| Gemini 3.5 Flash-Lite | %68 | %70 | %92 | %6 |

**Hakemler ayrıştığı için elle kontrol edildi.** Cevapların yaklaşık üçte birinde doğru/yanlış kararları farklı. Bunlardan 19'u madde metniyle karşılaştırıldı:
- Yaklaşık 16'sında Gemini haklı çıktı. Qwen hakemi, askerlik maddesini her işten çıkarmaya uygulayan ya da hastalık ödeneğini işsizlik parasıyla karıştıran cevapları geçirdi; bazen sorunun kendisini iddia saydı.
- Gemini de kendi ailesini biraz kayırıyor: öğrencide yanlış saydığı aynı hatayı yapan bir Flash-Lite cevabını geçirdi.

**Çıkarımlar**
- Burada güvenilecek hakem Gemini 3.5 Flash; yerel hakem kendi öğrencisine karşı fazla hoşgörülü.
- İçerikte eğitilmiş 4B model Gemini ile başa baş **değil**: yaklaşık %42'ye karşı %83 doğru (farkın bir kısmı hakemin aile kayırmasından). Başlıca hataları bir maddeyi yanlış duruma uygulamak ve süre gibi temel koşulları atlamak.
- Yine de fine-tuning çok işe yarıyor: iki hakem de her eğitilmiş modeli kendi temel modelinin üstünde görüyor.

**Kalite kapısı.** Çıktılar ve hakem kararları commit'lenir; `tests/test_quality.py`, kullanılan modelin atıf ya da hakem puanları kayıtlı seviyenin altına düşerse veya çıktıları yeniden puanlanmadan değişirse CI'ı durdurur.

## 🧪 Testler

**141 test, hepsi yaklaşık iki saniyede geçiyor.** GPU, model indirme ya da API anahtarı gerekmez: LLM yerine senaryolu sahte bir model, BGE-M3 yerine elle yazılmış küçük vektörler kullanılır.

| Alan | Test | Neyi doğruluyor |
|---|:---:|---|
| 📥 Veri çekme | 7 | Kanunlar doğru madde sınırlarından bölünüyor, metadata, 4 kanunun hepsi var, tekrar yok, TBK'da yalnızca kira bölümü var |
| 🔎 Arama | 11 | Dense anlama göre, sparse ortak kelimeye göre sıralıyor, hybrid ikisini birleştiriyor, kanun filtresi, reranker, Türkçe `İ`/`ı` küçültme |
| 📊 Ölçüm | 42 | Recall@k ve MRR hesabı, 39 test sorusunun her biri gerçekten var olan bir maddeyi gösteriyor |
| 🤖 Agent | 8 | Önce arıyor sonra cevaplıyor, olmayan madde modele bildiriliyor, araç döngüsü sınırlı, sağlayıcı ayarları |
| 🌐 API ve demo sayfası | 17 | Bütün uç noktalar, girdi doğrulama, LLM hatalarının HTTP kodlarına çevrilmesi, Türkçe ret mesajı, demo sayfası ve `/info` |
| 🔌 MCP sunucusu | 6 | Araçlar salt okunur, kanun filtresi, hatalı girdi ve olmayan madde araç hatası olarak dönüyor |
| 🧑‍🏫 Fine-tuning verisi | 10 | Soru ve cevap ayrıştırma, verilmeyen maddeye atıf yapan cevaplar atılıyor, üç GPU aşaması dosyalar üzerinden veri aktarıyor |
| 🎓 Eğitim ve karşılaştırma | 13 | Kayıp yalnızca cevapta, eğitim ve değerlendirme setleri madde paylaşmıyor, tüm karşılaştırma ölçütleri, rapor |
| 🏠 Yerel model | 9 | Thinking kapalı eğitim prompt'u ve JSON şeması, verilmeyen maddelere atıf atılıyor, araçsız agent yolu, sunucu hataları, vLLM komutu |
| 🧑‍⚖️ Hakem ve kalite kapısı | 14 | Geçersiz cevaplar hakeme gitmiyor, karar yalnızca aynı çıktı ve aynı hakem için yeniden kullanılıyor, toplu işleme, puanlar; kullanılan model kayıtlı puanlarının üstünde kalıyor |
| 🐳 Docker | 4 | vLLM konteyneri `serve.py` ile aynı bayrakları alıyor, API ona servis adıyla ulaşıyor, GPU servisi isteğe bağlı, API imajı bütün CUDA paketlerini dışarıda bırakıyor |

```bash
uv run pytest
```

## 🚀 Hızlı başlangıç

```bash
uv sync
export GEMINI_API_KEY=...   # aistudio.google.com'dan ücretsiz anahtar
uv run python -m turkish_legal_assistant.vectors   # bir kerelik: maddeleri vektöre çevirir (BGE-M3'ü indirir, ~2.3 GB)
uv run uvicorn turkish_legal_assistant.main:app --reload
```

Demo sayfası için http://localhost:8000 adresini açın (soru sorun, dayanılan maddeleri okuyun); API için http://localhost:8000/docs.

<details>
<summary>Diğer komutlar</summary>

```bash
export LLM_PROVIDER=claude ANTHROPIC_API_KEY=...   # Gemini yerine Claude
uv run python -m turkish_legal_assistant.ingest     # kanunları yeniden indirip böl
uv run python -m turkish_legal_assistant.evaluate   # arama yöntemlerini ölç
uv run pytest                                       # testler (model ya da API gerekmez)
uv run python -m turkish_legal_assistant.judge report --judge gemini   # cevap kalitesi puanları
```

Yerel model (NVIDIA GPU, 12 GB):

```bash
uv run python -m turkish_legal_assistant.finetune --size 4b      # adaptörü eğit (~52 dk)
uv run python -m turkish_legal_assistant.serve --size 4b         # 8001 portunda vLLM sunucusu
LLM_PROVIDER=local EMBED_DEVICE=cpu uv run uvicorn turkish_legal_assistant.main:app
```

</details>

## 🐳 Docker

API ve demo sayfası tek konteynerde çalışır. Eğitilmiş öğrenci model ise ayrı, GPU kullanan bir vLLM konteynerinde çalışır ve yalnızca `local` profiliyle başlar.

```bash
uv run python -m turkish_legal_assistant.vectors   # bir kerelik, bilgisayarda: embedding önbelleği data/'dan bağlanır
cp .env.example .env                               # GEMINI_API_KEY girin (ya da LLM_PROVIDER=claude / local)
docker compose up --build                          # API + demo arayüzü: http://localhost:8000
```

Yerel modelle (NVIDIA GPU + [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/), `models/` altında eğitilmiş adaptör):

```bash
# .env: LLM_PROVIDER=local
docker compose --profile local up --build          # önce vLLM açılır; API, vLLM hazır olana kadar bekler
```

`data/`, `models/` ve Hugging Face önbelleği kopyalanmaz, bağlanır (mount). Böylece modeller tekrar indirilmez.

| İmaj | Boyut | İçinde ne var |
|---|:---:|---|
| `api` | 2,6 GB | FastAPI, agent ve CPU'da çalışan BGE-M3. torch, kilit dosyasındaki sürümle CPU deposundan, CUDA paketleri olmadan kurulur (önceden 17,5 GB) |
| `vllm` | 35 GB | Resmi vLLM imajı ve bitsandbytes; Qwen3-4B'yi 4 bit olarak adaptörle birlikte sunar |

**Test edildi:** `local` profiliyle, Docker Desktop üzerinde (WSL 2, RTX 4070 Ti). İki imaj da build oluyor, vLLM GPU'da hazır duruma geçiyor, ardından API açılıyor ve demo sayfası eğitilmiş modelden cevap veriyor. İlk soru yaklaşık 20 saniye sürüyor (BGE-M3 yükleniyor), sonrakiler 3 ile 6 saniye. İki konteyner 12 GB GPU belleğinin yaklaşık 10,5 GB'ını kullanıyor. Bu yerel bir kurulumdur: bir sunucuya deploy edilmedi; sayfada giriş ya da istek sınırı yok.

## 🌐 API

| Metot | Yol | Açıklama |
|---|---|---|
| `GET` | `/` | Demo sayfası |
| `GET` | `/health` | Sağlık kontrolü |
| `GET` | `/info` | `/ask`'e hangi modelin cevap verdiği |
| `GET` | `/laws` | Kanunları listeler |
| `GET` | `/laws/{law}/articles/{article}` | Tek madde, örn. `/laws/4857/articles/53` |
| `POST` | `/ask` | `{"question": "Yıllık izin kaç gündür?"}` |

## 🔌 MCP sunucusu

Her MCP istemcisine üç salt okunur araç sunar: `search_articles`, `get_article`, `list_laws`. Düşünme işini istemcinin kendi modeli yaptığı için API anahtarı gerekmez.

**Claude Code**

```bash
claude mcp add turkish-legal -- uv run --directory /path/to/turkish-legal-assistant python -m turkish_legal_assistant.mcp_server
```

**Claude Desktop** (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "turkish-legal": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/turkish-legal-assistant", "python", "-m", "turkish_legal_assistant.mcp_server"]
    }
  }
}
```

## 📁 Proje yapısı

```
src/turkish_legal_assistant/
├── ingest.py        # kanunları indir ve böl
├── vectors.py       # BGE-M3 embedding + Qdrant arama
├── agent.py         # LangGraph agent
├── main.py          # FastAPI uygulaması
├── mcp_server.py    # MCP sunucusu
├── evaluate.py      # Recall@k / MRR
├── finetune_data.py # öğretmen modelle eğitim verisi üretimi
├── finetune.py      # QLoRA eğitimi
├── compare.py       # değerlendirme setinde model karşılaştırma
├── serve.py         # eğitilmiş model için vLLM sunucusu
├── local.py         # LLM_PROVIDER=local
├── judge.py         # LLM-as-a-judge
└── static/index.html # demo sayfası (düz HTML + JS, derleme adımı yok)
data/
├── articles.json    # 348 madde
├── golden.json      # 39 test sorusu
└── finetune/        # eğitim verisi, karşılaştırma çıktıları, hakem kararları
tests/               # 141 pytest testi, GPU ya da API anahtarı gerekmez
Dockerfile           # API imajı (yalnızca CPU torch; vLLM / eğitim paketleri yok)
docker/vllm.Dockerfile # vLLM imajı + bitsandbytes
compose.yaml         # api + isteğe bağlı vllm (profil: local)
```

---

<div align="center">

⚠️ **Bu proje genel bilgi verir, hukuki tavsiye değildir.**

Kod: [MIT Lisansı](LICENSE) · Kanun metinleri: [mevzuat.gov.tr](https://www.mevzuat.gov.tr)

</div>
