"""Build the Phase 6 fine-tuning set: a teacher model writes questions and answers, the student learns the answers.

Pipeline (runs on a GPU, see notebooks/finetune.ipynb):
  1. The teacher writes everyday questions for each article.
  2. Our retriever finds articles for each question, as /ask would.
  3. The teacher answers from those articles as LegalAnswer JSON.
  4. Answers citing articles that were not provided are dropped.
The student sees exactly the teacher's input, so it learns the answering step of RAG, not the law itself.
`generate` is any function from a batch of chat messages to texts (vLLM on Colab, a fake in tests).
"""

import json
import re
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from .models import Article, LegalAnswer
from .vectors import article_text

Messages = list[dict[str, str]]
Generate = Callable[[list[Messages]], list[str]]
Retrieve = Callable[[str], list[Article]]

QUESTIONS_PER_ARTICLE = 2
EVAL_EVERY = 10  # every 10th article goes to the eval split, so eval questions are about unseen articles

QUESTION_PROMPT = """Below is an article of Turkish law. Write {n} different questions an ordinary citizen
might ask that this article answers. Write them in everyday Turkish, as the person would phrase their own
situation, without legal terms, law names or article numbers. One question per line, nothing else.

{article}"""

ANSWER_SYSTEM = """You are a legal information assistant for Turkish legislation.
Answer the question in Turkish, clearly and briefly, for a non-lawyer, using only the articles given.
Cite every article the answer relies on. If the articles do not answer the question, say so and cite nothing.
You give general information, not legal advice. Reply with JSON: {"answer": "...", "citations": [{"law_number": "...", "article_number": "..."}]}"""


def is_repealed(article: Article) -> bool:
    return article.text.lstrip().startswith("(Mülga")


def question_messages(article: Article, n: int = QUESTIONS_PER_ARTICLE) -> Messages:
    return [{"role": "user", "content": QUESTION_PROMPT.format(n=n, article=article_text(article))}]


def parse_questions(text: str, n: int = QUESTIONS_PER_ARTICLE) -> list[str]:
    # Models often number or bullet the lines despite the instruction.
    lines = (re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip() for line in text.splitlines())
    return [q for q in lines if q.endswith("?") and len(q) > 10][:n]


def answer_messages(question: str, articles: list[Article]) -> Messages:
    context = "\n\n".join(f"[{a.law_number}/{a.article_number}] {article_text(a)}" for a in articles)
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"Maddeler:\n{context}\n\nSoru: {question}"},
    ]


def parse_answer(text: str, articles: list[Article]) -> LegalAnswer | None:
    """Keep only well-formed answers whose citations are all among the given articles."""
    try:
        answer = LegalAnswer.model_validate_json(text)
    except ValidationError:
        return None
    given = {(a.law_number, a.article_number) for a in articles}
    if not answer.answer.strip() or any((c.law_number, c.article_number) not in given for c in answer.citations):
        return None
    return answer


def build_examples(articles: list[Article], generate: Generate, retrieve: Retrieve) -> dict[str, list[dict]]:
    """Returns {"train": [...], "eval": [...]}, each example {"messages": [...], "source": "law/article"}."""
    sources = [a for a in articles if not is_repealed(a)]
    question_texts = generate([question_messages(a) for a in sources])
    pairs = [(i, q) for i, text in enumerate(question_texts) for q in parse_questions(text)]

    contexts = [retrieve(q) for _, q in pairs]
    prompts = [answer_messages(q, ctx) for (_, q), ctx in zip(pairs, contexts)]
    answers = generate(prompts)

    splits: dict[str, list[dict]] = {"train": [], "eval": []}
    for (i, _), ctx, prompt, text in zip(pairs, contexts, prompts, answers):
        parsed = parse_answer(text, ctx)
        if parsed is None:
            continue
        source = sources[i]
        split = "eval" if i % EVAL_EVERY == 0 else "train"
        splits[split].append(
            {
                "messages": [*prompt, {"role": "assistant", "content": parsed.model_dump_json()}],
                "source": f"{source.law_number}/{source.article_number}",
            }
        )
    return splits


def save(splits: dict[str, list[dict]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in splits.items():
        lines = (json.dumps(e, ensure_ascii=False) for e in examples)
        (out_dir / f"{name}.jsonl").write_text("\n".join(lines) + "\n")
