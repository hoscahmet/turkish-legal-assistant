"""Build the Phase 6 fine-tuning set: a teacher model writes questions and answers, the student learns the answers.

Pipeline, one process per stage so the teacher (vLLM) and BGE-M3 never share the 12 GB GPU:
  1. questions: the teacher writes everyday questions for each article.
  2. retrieve:  our retriever finds articles for each question, as /ask would.
  3. answers:   the teacher answers from those articles as LegalAnswer JSON;
                answers citing articles that were not provided are dropped.
Each stage reads the previous stage's file in data/finetune/ and writes its own:
  uv run python -m turkish_legal_assistant.finetune_data questions|retrieve|answers
The student sees exactly the teacher's input, so it learns the answering step of RAG, not the law itself.
`generate` is any function from a batch of chat messages to texts (vLLM here, a fake in tests).
"""

import argparse
import json
import os
import re
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from .data import ARTICLES, find_article
from .ingest import DATA_DIR
from .models import Article, LegalAnswer
from .text import tr_lower
from .vectors import article_text

Messages = list[dict[str, str]]
Generate = Callable[[list[Messages]], list[str]]
Retrieve = Callable[[str], list[Article]]

QUESTIONS_PER_ARTICLE = 2
EVAL_EVERY = 10  # every 10th article goes to the eval split, so eval questions are about unseen articles
TOP_K = 5  # articles retrieved per question, the /ask default

OUT_DIR = DATA_DIR / "finetune"
QUESTIONS_FILE = OUT_DIR / "questions.jsonl"
CONTEXTS_FILE = OUT_DIR / "contexts.jsonl"

TEACHER = "Qwen/Qwen3-8B-AWQ"
# 8B AWQ weights take ~6 GB; the rest of the budget is KV cache. Longer prompts (5 very long articles) are skipped.
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.8

QUESTION_PROMPT = """Below is an article of Turkish law. Write {n} different questions an ordinary citizen
might ask that this article answers. Write them in everyday Turkish, as the person would phrase their own
situation, without legal terms, law names or article numbers. The person has never seen this article, so never
refer to it ("bu kanun", "bu madde"). One question per line, nothing else.

{article}"""

ANSWER_SYSTEM = """You are a legal information assistant for Turkish legislation.
Answer the question in Turkish, clearly and briefly, for a non-lawyer, using only the articles given.
Cite every article the answer relies on. If the articles do not answer the question, say so and cite nothing.
State only what the articles say; do not add rules from elsewhere. In the answer text, name an article as the
law's name and article number (e.g. "İş Kanunu madde 53"), never with bracketed codes like [4857/53].
You give general information, not legal advice. Reply with JSON: {"answer": "...", "citations": [{"law_number": "...", "article_number": "..."}]}"""


def is_repealed(article: Article) -> bool:
    return article.text.lstrip().startswith("(Mülga")


def question_messages(article: Article, n: int = QUESTIONS_PER_ARTICLE) -> Messages:
    return [{"role": "user", "content": QUESTION_PROMPT.format(n=n, article=article_text(article))}]


# A real user has no article in front of them, so a question pointing at one is a teacher artifact.
SEES_THE_ARTICLE = re.compile(r"\bbu (kanun|madde|yasa|hüküm|düzenleme)")


def parse_questions(text: str, n: int = QUESTIONS_PER_ARTICLE) -> list[str]:
    # Models often number or bullet the lines despite the instruction.
    lines = (re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip() for line in text.splitlines())
    return [q for q in lines if q.endswith("?") and len(q) > 10 and not SEES_THE_ARTICLE.search(tr_lower(q))][:n]


def strip_thinking(text: str) -> str:
    """Qwen3 in thinking mode writes <think>...</think> before the reply; the student learns only the reply."""
    return text.rsplit("</think>", 1)[-1].strip()


def answer_messages(question: str, articles: list[Article]) -> Messages:
    context = "\n\n".join(f"[{a.law_number}/{a.article_number}] {article_text(a)}" for a in articles)
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"Maddeler:\n{context}\n\nSoru: {question}"},
    ]


# Qwen sometimes drifts into Chinese mid-answer ("İş Kanunu'ndaki相关规定 include: ...").
FOREIGN_SCRIPT = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def parse_answer(text: str, articles: list[Article]) -> LegalAnswer | None:
    """Keep only well-formed answers whose citations are all among the given articles."""
    try:
        answer = LegalAnswer.model_validate_json(text)
    except ValidationError:
        return None
    given = {(a.law_number, a.article_number) for a in articles}
    if not answer.answer.strip() or FOREIGN_SCRIPT.search(answer.answer) or any((c.law_number, c.article_number) not in given for c in answer.citations):
        return None
    return answer


def ref(article: Article) -> str:
    return f"{article.law_number}/{article.article_number}"


def from_ref(r: str) -> Article:
    law_number, article_number = r.split("/", 1)
    return find_article(law_number, article_number)


def make_questions(articles: list[Article], generate: Generate) -> list[dict]:
    """Stage 1. Returns [{"source": "law/article", "split": "train"|"eval", "question": "..."}]."""
    sources = [a for a in articles if not is_repealed(a)]
    texts = generate([question_messages(a) for a in sources])
    return [
        {"source": ref(a), "split": "eval" if i % EVAL_EVERY == 0 else "train", "question": q}
        for i, (a, text) in enumerate(zip(sources, texts))
        for q in parse_questions(text)
    ]


def add_contexts(items: list[dict], retrieve: Retrieve) -> list[dict]:
    """Stage 2. Adds "context": the refs of the articles retrieved for each question."""
    return [{**item, "context": [ref(a) for a in retrieve(item["question"])]} for item in items]


def make_answers(items: list[dict], generate: Generate) -> dict[str, list[dict]]:
    """Stage 3. Returns {"train": [...], "eval": [...]}, each example {"messages": [...], "source": "law/article"}."""
    contexts = [[from_ref(r) for r in item["context"]] for item in items]
    prompts = [answer_messages(item["question"], ctx) for item, ctx in zip(items, contexts)]
    answers = generate(prompts)

    splits: dict[str, list[dict]] = {"train": [], "eval": []}
    for item, ctx, prompt, text in zip(items, contexts, prompts, answers):
        parsed = parse_answer(text, ctx)
        if parsed is None:
            continue
        splits[item["split"]].append(
            {"messages": [*prompt, {"role": "assistant", "content": parsed.model_dump_json()}], "source": item["source"]}
        )
    return splits


def build_examples(articles: list[Article], generate: Generate, retrieve: Retrieve) -> dict[str, list[dict]]:
    """All three stages in one process; fine for tests, too big for one 12 GB GPU."""
    return make_answers(add_contexts(make_questions(articles, generate), retrieve), generate)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the previous stage first")
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def save(splits: dict[str, list[dict]], out_dir: Path) -> None:
    for name, examples in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", examples)


def vllm_generate(
    thinking: bool,
    json_schema: dict | None = None,
    model: str = TEACHER,
    adapter: Path | None = None,
    quantization: str | None = None,
) -> Generate:
    """Loads a model once (the teacher by default); the returned function runs a whole batch through vLLM.

    `adapter` is a LoRA adapter directory served on top of `model` without merging (the fine-tuned student).
    `quantization="bitsandbytes"` loads an unquantized model in 4 bits, for models too big for the GPU in bf16.

    thinking=True lets Qwen3 reason in <think> before replying: slower, but without it the 8B teacher
    contradicted and embellished the articles in trial answers. The JSON schema then applies only after </think>.
    """
    # FlashInfer's sampler is JIT-compiled with nvcc, and WSL here has the CUDA driver but no toolkit.
    # PyTorch's own top-p/top-k sampler needs no compiler and is fast enough for offline batches.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams  # imported here: heavy, and only this machine's GPU stages need it
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(
        model,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        reasoning_parser="qwen3",  # tells structured outputs where the reasoning ends
        enable_lora=adapter is not None,
        max_lora_rank=16,  # the rank finetune.py trains with
        quantization=quantization,
    )
    lora = LoRARequest("adapter", 1, str(adapter)) if adapter else None
    params = SamplingParams(
        # Qwen3's recommended settings for each mode
        temperature=0.6 if thinking else 0.7,
        top_p=0.95 if thinking else 0.8,
        top_k=20,
        max_tokens=3072 if thinking else 1024,
        structured_outputs=StructuredOutputsParams(json=json_schema) if json_schema else None,
    )
    tokenizer = llm.get_tokenizer()

    def fits(messages: Messages) -> bool:
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=thinking)
        return len(ids) + params.max_tokens <= MAX_MODEL_LEN

    def generate(batch: list[Messages]) -> list[str]:
        keep = [i for i, m in enumerate(batch) if fits(m)]
        if len(keep) < len(batch):
            print(f"Skipping {len(batch) - len(keep)} prompts longer than {MAX_MODEL_LEN} tokens")
        outputs = llm.chat(
            [batch[i] for i in keep], params, lora_request=lora, chat_template_kwargs={"enable_thinking": thinking}
        )
        texts = [""] * len(batch)  # skipped prompts get "", which the parsers drop
        for i, out in zip(keep, outputs):
            texts[i] = strip_thinking(out.outputs[0].text)
        return texts

    return generate


def dense_retrieve(question: str) -> list[Article]:
    # Not agent.retrieve: that silently falls back to keyword search if the embedding cache is missing.
    from .vectors import get_index

    return [a for a, _ in get_index().search(question, TOP_K, "dense")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["questions", "retrieve", "answers"])
    parser.add_argument("--limit", type=int, help="use only the first N articles (quick trial run)")
    args = parser.parse_args()

    if args.stage == "questions":
        articles = ARTICLES[: args.limit] if args.limit else ARTICLES
        rows = make_questions(articles, vllm_generate(thinking=False))  # writing questions needs no reasoning
        write_jsonl(QUESTIONS_FILE, rows)
        print(f"{len(rows)} questions -> {QUESTIONS_FILE}")
    elif args.stage == "retrieve":
        rows = add_contexts(read_jsonl(QUESTIONS_FILE), dense_retrieve)
        write_jsonl(CONTEXTS_FILE, rows)
        print(f"{len(rows)} questions with context -> {CONTEXTS_FILE}")
    else:
        items = read_jsonl(CONTEXTS_FILE)
        splits = make_answers(items, vllm_generate(thinking=True, json_schema=LegalAnswer.model_json_schema()))
        save(splits, OUT_DIR)
        kept = sum(len(v) for v in splits.values())
        print(f"Kept {kept}/{len(items)} answers: train {len(splits['train'])}, eval {len(splits['eval'])} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
