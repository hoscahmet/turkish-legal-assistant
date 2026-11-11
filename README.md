<div align="center">

# ⚖️ Turkish Legal Assistant

**Ask questions about Turkish law in plain Turkish, and get short answers that cite the exact articles.**

[![English](https://img.shields.io/badge/🇬🇧_English-selected-1f6feb?style=for-the-badge)](README.md)
[![Türkçe](https://img.shields.io/badge/🇹🇷_Türkçe-oku-e30a17?style=for-the-badge)](README.tr.md)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-agent-1C3C3C?logo=langchain&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-vector_search-DC244C)
![BGE-M3](https://img.shields.io/badge/BGE--M3-embeddings-FFD21E?logo=huggingface&logoColor=black)
![MCP](https://img.shields.io/badge/MCP-server-8A2BE2)
![vLLM](https://img.shields.io/badge/vLLM-QLoRA_student-30A2FF)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![uv](https://img.shields.io/badge/uv-package_manager-DE5FE9)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 💬 Example

> **Question:** *Ev sahibi oğlu oturacak diye beni evden çıkarabilir mi?*
> (Can my landlord evict me because his son will move in?)
>
> **Answer:** Yes, but only through a court case, and only if his son genuinely needs to live there. If he evicts you this way, he cannot rent the flat to anyone else for 3 years without a valid reason.
>
> 📎 **Sources:** Turkish Code of Obligations, art. 350 · art. 355

Every answer comes from the law text itself, not from the model's memory. Citations to articles that do not exist are removed automatically.

## ✨ What it does

| | |
|---|---|
| 📚 **4 laws, 348 articles** | Labour Law (4857), Personal Data Protection (6698, KVKK), Consumer Protection (6502), Code of Obligations: lease section (6098) |
| 🔎 **Semantic search** | Finds "işveren ücreti ödemiyor" when you ask "patron maaş vermiyor" |
| 🤖 **Agent** | The model searches, reads, searches again if needed, then answers |
| 🔌 **MCP server** | Plug the search into Claude Desktop, Claude Code or any MCP client |
| 🛡️ **Hallucination guard** | Only real, existing articles are shown as sources |
| 🔄 **Any LLM** | Gemini (free tier), Claude, or a fine-tuned local Qwen3-4B, switched with one setting |
| 🖥️ **Demo page** | Ask in the browser and read the cited articles; the page shows which model answered |
| 🐳 **Docker Compose** | One command starts the API and demo page; a second, GPU-backed container serves the local model |

## 🏗️ How it works

```mermaid
flowchart LR
    Q(["❓ Question"]) --> A{"🤖 Agent"}
    A -- search --> S["🔎 Dense search<br/>BGE-M3 + Qdrant"]
    S --> D[("📚 348 articles")]
    D --> A
    A -- enough info --> R["✍ Answer<br/>JSON + citations"]
    R --> G["🛡 Citation check"]
    G --> OUT(["✅ Answer + sources"])

    classDef blue fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a
    classDef green fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef yellow fill:#fef9c3,stroke:#a16207,color:#713f12
    classDef purple fill:#f3e8ff,stroke:#7e22ce,color:#581c87
    class Q,OUT green
    class A purple
    class S,D blue
    class R,G yellow
```

This pattern is called **RAG** (Retrieval-Augmented Generation): first *find* the right articles, then let the model *write* the answer from them only.

## 🗺️ Roadmap

| Phase | What | Status |
|:---:|---|:---:|
| 0 | FastAPI skeleton, Pydantic models | ✅ |
| 1 | Scrape mevzuat.gov.tr, split into articles with metadata | ✅ |
| 2 | LLM with tool calling and structured output | ✅ |
| 3 | BGE-M3 embeddings, Qdrant, hybrid search, reranker, evaluation | ✅ |
| 4 | LangGraph agent | ✅ |
| 5 | MCP server | ✅ |
| 6 | Fine-tune a small model with QLoRA, serve with vLLM | ✅ |
| 7 | LLM-as-a-judge, CI quality gate | ✅ |
| 7+ | RAGAS, Langfuse tracing | ⏳ |
| 8 | Docker Compose, demo UI | ✅ |

## 📊 Retrieval results

Measured on 39 everyday Turkish questions with known correct articles.

| Retriever | Recall@1 | Recall@5 | MRR |
|---|:---:|:---:|:---:|
| Keyword overlap (baseline) | 0.19 | 0.38 | 0.30 |
| BGE-M3 sparse | 0.33 | 0.45 | 0.42 |
| BGE-M3 hybrid (RRF) | 0.38 | **0.87** | 0.61 |
| 🏆 **BGE-M3 dense** (used) | **0.64** | 0.86 | **0.77** |
| Dense + bge-reranker-v2-m3 | 0.56 | 0.86 | 0.72 |

**Takeaways**
- Dense search beats keywords by a wide margin (MRR 0.77 vs 0.30).
- Hybrid fusion and the reranker did *not* help on this data. The reranker tends to push long, general articles above the specific answer.
- With 39 questions, a 0.05 MRR gap is about two questions, so small differences are not conclusive.

## 🎓 Local model (fine-tuning)

The answer step can also run on a small open model on one consumer GPU (RTX 4070 Ti, 12 GB), with no API at all.

1. **Teacher data.** Qwen3-8B (AWQ, served with vLLM) answers generated questions from the retrieved articles: 460 training and 53 evaluation examples. Articles behind the evaluation questions never appear in training. Only Apache-2.0 models write the data; no Claude or Gemini output is used for training.
2. **QLoRA.** Qwen3-4B is loaded in 4 bits and frozen. Small LoRA adapters (0.8% of the weights) learn to turn *question + articles* into the answer JSON. The loss is computed on the answer only. Training takes 52 minutes.
3. **Serving.** vLLM serves the base model with the adapter on top, as an OpenAI-compatible server. With `LLM_PROVIDER=local` the API retrieves the articles and sends them to it in one step, with the JSON schema enforced.

Measured on the 53 evaluation questions, every model getting the same prompt:

| Model | Valid JSON | Grounded | Right source | Agreement | s/question |
|---|:---:|:---:|:---:|:---:|:---:|
| Qwen3-1.7B, base | 21% | 0% | 0% | 0.00 | 0.63 |
| Qwen3-1.7B, fine-tuned | 100% | 96% | 69% | 0.64 | 0.50 |
| Qwen3-4B, base | 100% | 62% | 49% | 0.40 | 0.94 |
| 🏆 **Qwen3-4B, fine-tuned** (used) | 100% | **100%** | 77% | **0.70** | 0.92 |
| Qwen3-4B, fine-tuned 1 epoch | 100% | 100% | 77% | 0.67 | 1.40 |
| Gemini 3.5 Flash-Lite | 100% | 100% | 77% | 0.68 | 6.24 |
| Qwen3-8B teacher | 100% | 98% | 82% | 0.87 | 2.38 |

*Grounded:* every citation is one of the articles given. *Right source:* cites the article the question was written from, when it was retrieved. *Agreement:* citation F1 against the teacher's answer (so the teacher scores high by construction).

**Takeaways**
- Fine-tuning turns a 4B model that cites made-up articles into one that matches Gemini on these citation scores, running locally. On content it does not; see the next section.
- One epoch gave a lower evaluation loss (0.390 vs 0.405) but not better answers, so the 3-epoch adapter is kept.
- These scores check citations, not legal content. The student can still get a fact wrong (e.g. it missed the CPI cap on rent increases in TBK 344); judging content is the next section.

## 🧑‍⚖️ Answer quality (LLM-as-a-judge)

The citation scores above cannot tell a right answer from a wrong one that cites the right article. So a judge model reads each answer with the articles it was given, splits it into claims and labels every claim **supported**, **contradicted** (an article says otherwise, or the answer says the articles do not cover something they do) or **unsupported**. It also says whether the answer is **complete**: does it give the rule, number or condition the question needs?

*Correct:* complete with no contradicted claim. *Faithful:* every claim supported. *Contradicted:* at least one claim the articles contradict.

**Judge: Gemini 3.5 Flash** (independent of the Qwen models; stronger than the judged Flash-Lite)

| Model | Correct | Complete | Faithful | Contradicted |
|---|:---:|:---:|:---:|:---:|
| Qwen3-1.7B, fine-tuned | 34% | 43% | 47% | 43% |
| Qwen3-4B, fine-tuned (used) | 42% | 47% | 58% | 34% |
| Qwen3-4B, fine-tuned 1 epoch | 34% | 42% | 47% | 38% |
| Qwen3-8B teacher | 58% | 68% | 60% | 28% |
| Gemini 3.5 Flash-Lite | **83%** | **83%** | **100%** | **0%** |

The base models were not judged by Gemini (the run was stopped for cost); see the local judge below.

**Judge: Qwen3-8B** (local and free, but it wrote the student's training data)

| Model | Correct | Complete | Faithful | Contradicted |
|---|:---:|:---:|:---:|:---:|
| Qwen3-1.7B, base | 9% | 9% | 19% | 2% |
| Qwen3-1.7B, fine-tuned | 38% | 38% | 53% | 30% |
| Qwen3-4B, base | 51% | 51% | 79% | 11% |
| Qwen3-4B, fine-tuned (used) | 68% | 72% | 74% | 17% |
| Qwen3-4B, fine-tuned 1 epoch | 58% | 60% | 81% | 9% |
| Qwen3-8B teacher | 81% | 81% | 81% | 8% |
| Gemini 3.5 Flash-Lite | 68% | 70% | 92% | 6% |

**The judges disagree, so they were checked by hand.** They gave a different correct/incorrect verdict on about a third of the answers. 19 of those were checked against the article text:
- Gemini was right in about 16. The Qwen judge passed answers that apply the military-service article to any dismissal or confuse sick pay with unemployment pay, and it sometimes labels the question itself as a claim.
- Gemini also favours its own family a little: it passed a Flash-Lite answer making the same misapplication it failed in the student.

**Takeaways**
- Gemini 3.5 Flash is the judge to trust here; the local judge is too lenient on its own student.
- On content, the fine-tuned 4B does **not** match Gemini: about 42% correct against 83% (some of that gap is the judge's family bias). Its main faults are applying an article to the wrong situation and leaving out key conditions such as deadlines.
- Fine-tuning still helps a lot: both judges rank every fine-tuned model above its base model.

**Quality gate.** The outputs and verdicts are committed, and `tests/test_quality.py` fails CI if the served model's citation or judge scores drop below the recorded level, or if its outputs change without being judged again.

## 🧪 Tests

**141 tests, all passing in about two seconds.** No GPU, model download or API key is needed: the LLM is a scripted fake, and BGE-M3 is replaced by tiny hand-made vectors.

| Area | Tests | What they check |
|---|:---:|---|
| 📥 Ingest & data | 7 | Laws split at the right article boundaries, metadata, all 4 laws present, no duplicates, TBK holds only the lease section |
| 🔎 Search | 11 | Dense ranks by meaning, sparse by shared words, hybrid fuses both, law filter, reranker, Turkish `İ`/`ı` lowercasing |
| 📊 Evaluation | 42 | Recall@k and MRR math, and every one of the 39 golden questions points to an article that exists |
| 🤖 Agent | 8 | Searches then answers, unknown articles reported back to the model, tool loop capped, provider settings |
| 🌐 API & demo page | 17 | Every endpoint, input validation, LLM errors mapped to HTTP codes, Turkish refusal message, the demo page and `/info` |
| 🔌 MCP server | 6 | Tools are read-only, law filter, invalid input and missing articles become tool errors |
| 🧑‍🏫 Fine-tuning data | 10 | Question and answer parsing, citations to articles not given are dropped, the three GPU stages round-trip through files |
| 🎓 Training & comparison | 13 | Loss only on the answer, train/eval splits share no articles, every comparison score, report |
| 🏠 Local model | 9 | Training prompt with thinking off and JSON schema, citations not given are dropped, no-tool agent path, server errors, vLLM command |
| 🧑‍⚖️ Judge & quality gate | 14 | Invalid answers skip the judge, verdicts reused only for the same output and judge, batching, scores; the served model stays above its recorded scores |
| 🐳 Docker | 4 | The vLLM container gets the same flags as `serve.py`, the API reaches it by service name, the GPU service is opt-in, the API image drops every CUDA wheel |

```bash
uv run pytest
```

## 🚀 Quick start

```bash
uv sync
export GEMINI_API_KEY=...   # free key from aistudio.google.com
uv run python -m turkish_legal_assistant.vectors   # one-time: embed articles (downloads BGE-M3, ~2.3 GB)
uv run uvicorn turkish_legal_assistant.main:app --reload
```

Open http://localhost:8000 for the demo page (ask a question, read the cited articles), or http://localhost:8000/docs for the API.

<details>
<summary>More commands</summary>

```bash
export LLM_PROVIDER=claude ANTHROPIC_API_KEY=...   # use Claude instead of Gemini
uv run python -m turkish_legal_assistant.ingest     # re-download and re-split the laws
uv run python -m turkish_legal_assistant.evaluate   # score the retrievers
uv run pytest                                       # tests (no model or API needed)
uv run python -m turkish_legal_assistant.judge report --judge gemini   # answer-quality scores
```

Local model (NVIDIA GPU, 12 GB):

```bash
uv run python -m turkish_legal_assistant.finetune --size 4b      # train the adapter (~52 min)
uv run python -m turkish_legal_assistant.serve --size 4b         # vLLM server on port 8001
LLM_PROVIDER=local EMBED_DEVICE=cpu uv run uvicorn turkish_legal_assistant.main:app
```

</details>

## 🐳 Docker

The API and demo page run in one container; the fine-tuned student runs in a second, GPU-backed vLLM container that only starts with the `local` profile.

```bash
uv run python -m turkish_legal_assistant.vectors   # once, on the host: the embedding cache is mounted from data/
cp .env.example .env                               # set GEMINI_API_KEY (or LLM_PROVIDER=claude / local)
docker compose up --build                          # API + demo UI on http://localhost:8000
```

With the local model (NVIDIA GPU + [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/), trained adapter in `models/`):

```bash
# .env: LLM_PROVIDER=local
docker compose --profile local up --build          # vLLM starts first; the API waits until it is healthy
```

`data/`, `models/` and the Hugging Face cache are mounted, not copied, so models are not downloaded again.

| Image | Size | What is inside |
|---|:---:|---|
| `api` | 2.6 GB | FastAPI, the agent and BGE-M3 on the CPU. torch is installed from the CPU index at the locked version, without its CUDA wheels (17.5 GB before) |
| `vllm` | 35 GB | The official vLLM image plus bitsandbytes, serving Qwen3-4B in 4 bits with the adapter |

**Tested** with the `local` profile on Docker Desktop (WSL 2, RTX 4070 Ti): both images build, vLLM turns healthy on the GPU, then the API starts and the demo page answers from the fine-tuned model. The first question takes about 20 seconds (BGE-M3 loads), later ones 3 to 6. The two containers use about 10.5 GB of the 12 GB of GPU memory. This is a local setup: nothing is deployed to a server, and the page has no login or rate limit.

## 🌐 API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Demo page |
| `GET` | `/health` | Health check |
| `GET` | `/info` | Which LLM answers `/ask` |
| `GET` | `/laws` | List the laws |
| `GET` | `/laws/{law}/articles/{article}` | One article, e.g. `/laws/4857/articles/53` |
| `POST` | `/ask` | `{"question": "Yıllık izin kaç gündür?"}` |

## 🔌 MCP server

Gives any MCP client three read-only tools: `search_articles`, `get_article`, `list_laws`. The client's own model does the reasoning, so no API key is needed.

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

## 📁 Project structure

```
src/turkish_legal_assistant/
├── ingest.py        # download and split the laws
├── vectors.py       # BGE-M3 embeddings + Qdrant search
├── agent.py         # LangGraph agent
├── main.py          # FastAPI app
├── mcp_server.py    # MCP server
├── evaluate.py      # Recall@k / MRR
├── finetune_data.py # teacher-generated training data
├── finetune.py      # QLoRA training
├── compare.py       # model comparison on the eval split
├── serve.py         # vLLM server for the fine-tuned model
├── local.py         # LLM_PROVIDER=local
├── judge.py         # LLM-as-a-judge
└── static/index.html # demo page (plain HTML + JS, no build step)
data/
├── articles.json    # 348 articles
├── golden.json      # 39 test questions
└── finetune/        # training data, comparison outputs, judge verdicts
tests/               # 141 pytest tests, no GPU or API key needed
Dockerfile           # API image (CPU-only torch, no vLLM / training stack)
docker/vllm.Dockerfile # vLLM image + bitsandbytes
compose.yaml         # api + optional vllm (profile: local)
```

---

<div align="center">

⚠️ **This project gives general information, not legal advice.**

Code: [MIT License](LICENSE) · Law texts: [mevzuat.gov.tr](https://www.mevzuat.gov.tr)

</div>
