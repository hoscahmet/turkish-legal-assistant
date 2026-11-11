"""LLM_PROVIDER=local: the fine-tuned student, served by vLLM's OpenAI-compatible server.

The student learned only the answering step of RAG, not tool calling, so there is no agent loop here:
    retrieve top_k articles -> the exact prompt it was trained on -> LegalAnswer JSON
The served adapter (4B, trained on the verified set) reasons in <think> before it answers: on the eval questions
that took it from 65% to 80% correct by the local judge, for about one more second per question. The adapters
trained on the first set gain nothing from thinking; serve those with LOCAL_THINKING=0.
Start the server first (another terminal): uv run python -m turkish_legal_assistant.serve [--size 4b]
"""

import os
from collections.abc import Callable

import httpx
from pydantic import ValidationError

from .finetune_data import FOREIGN_SCRIPT, Messages, answer_messages
from .models import Article, AskResponse, LegalAnswer

LOCAL_URL = os.environ.get("LOCAL_LLM_URL", "http://localhost:8001/v1")
SERVED_NAME = "legal"  # the LoRA adapter's name on the server (serve.py --lora-modules)
THINKING = os.environ.get("LOCAL_THINKING", "1") != "0"
Post = Callable[[str, dict], dict]


class LocalModelUnavailable(Exception):
    pass


class LocalModelError(Exception):
    pass


def chat_payload(messages: Messages) -> dict:
    return {
        "model": SERVED_NAME,
        "messages": messages,
        # Qwen3's recommended sampling for each mode, as in compare.py (finetune_data.vllm_generate).
        "temperature": 0.6 if THINKING else 0.7,
        "top_p": 0.95 if THINKING else 0.8,
        "top_k": 20,
        "max_tokens": 3072 if THINKING else 1024,
        "chat_template_kwargs": {"enable_thinking": THINKING},
        # vLLM constrains decoding to this schema, so the reply is always parseable JSON. The server's reasoning
        # parser (serve.py) keeps the thinking out of `content` and applies the schema only after it.
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "LegalAnswer", "schema": LegalAnswer.model_json_schema()},
        },
    }


def http_post(url: str, payload: dict) -> dict:
    try:
        response = httpx.post(url, json=payload, timeout=120)
    except httpx.TransportError as e:
        raise LocalModelUnavailable(f"no local model server at {LOCAL_URL} ({e}); start serve.py") from e
    if response.is_error:
        raise LocalModelError(f"local model server returned {response.status_code}: {response.text[:300]}")
    return response.json()


def answer(question: str, articles: list[Article], post: Post = http_post) -> AskResponse:
    payload = chat_payload(answer_messages(question, articles))
    text = post(f"{LOCAL_URL}/chat/completions", payload)["choices"][0]["message"]["content"]
    if FOREIGN_SCRIPT.search(text):
        # Qwen now and then drifts into Chinese mid-sentence (1 or 2 of 53 eval answers); sampling again usually fixes it.
        text = post(f"{LOCAL_URL}/chat/completions", payload)["choices"][0]["message"]["content"]
    try:
        parsed = LegalAnswer.model_validate_json(text)
    except ValidationError as e:
        raise LocalModelError(f"local model returned invalid JSON: {text[:300]}") from e
    # Hallucination guard: keep only citations of articles the model was actually given, once each.
    given = {(a.law_number, a.article_number): a for a in articles}
    keys = dict.fromkeys((c.law_number, c.article_number) for c in parsed.citations)
    return AskResponse(question=question, answer=parsed.answer, sources=[given[k] for k in keys if k in given])
