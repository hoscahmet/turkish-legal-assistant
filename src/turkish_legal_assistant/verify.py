"""Build a cleaner, larger training set (data/finetune/v2/): the teacher's answers are checked before the student sees them.

The first set (finetune_data.py) took every well-formed teacher answer. Judged by the same local judge as in judge.py,
about a quarter of those answers were incomplete, unsupported or contradicted by the articles. Here an answer goes
into training only if the judge finds it complete with every claim supported, and it passes two style checks.

Stages, one process each (the teacher, BGE-M3 and the judge do not share the 12 GB GPU):
  questions  teacher: more questions for every training article (never for the eval articles)
  retrieve   BGE-M3: the context for each new question, as /ask would retrieve it
  answers    teacher: answers the new questions
  judge      local judge: a verdict for every original and new answer, and for the repair candidates once they exist
  repair     teacher: fresh candidate answers for each original answer the judge flagged
  build      writes v2/train.jsonl: clean originals, the first clean candidate of each flagged one, clean new answers
Run them as questions, retrieve, answers, judge, repair, judge, build:
  uv run python -m turkish_legal_assistant.verify <stage>
The eval split is untouched, so results stay comparable with the first adapter.
"""

import argparse
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from . import judge as jd
from .compare import given_articles
from .finetune_data import (
    CONTEXTS_FILE,
    FOREIGN_SCRIPT,
    OUT_DIR,
    Generate,
    Messages,
    add_contexts,
    dense_retrieve,
    from_ref,
    make_answers,
    parse_questions,
    read_jsonl,
    write_jsonl,
)
from .text import tr_lower
from .vectors import article_text

V2_DIR = OUT_DIR / "v2"
WORK_DIR = V2_DIR / "work"  # intermediate files, gitignored
EXTRA_QUESTIONS = 3  # new questions asked for per training article
CANDIDATES = 3  # fresh answers per flagged original

EXTRA_PROMPT = """Below is an article of Turkish law. Write {n} different questions an ordinary citizen
might ask that this article answers. Write them in everyday Turkish, as the person would phrase their own
situation, without legal terms, law names or article numbers. The person has never seen this article, so never
refer to it ("bu kanun", "bu madde"). Each question must be understandable on its own and must be about a
different rule, number or condition in the article than these existing questions:
{existing}
One question per line, nothing else.

{article}"""

# "[4857/53]" or "4857/53" in the answer text: the prompt tells the model to write "İş Kanunu madde 53" instead.
BRACKET_CODE = re.compile(r"\[\d{4}/[^\]]+\]|\b\d{4}/\d+\b")


def extra_question_messages(source: str, existing: list[str], n: int = EXTRA_QUESTIONS) -> Messages:
    listed = "\n".join(f"- {q}" for q in existing)
    return [{"role": "user", "content": EXTRA_PROMPT.format(n=n, existing=listed, article=article_text(from_ref(source)))}]


def make_extra_questions(items: list[dict], generate: Generate, n: int = EXTRA_QUESTIONS) -> list[dict]:
    """New question rows for the training articles in `items` (rows of contexts.jsonl); eval rows are ignored."""
    by_source: dict[str, list[str]] = {}
    for item in items:
        if item["split"] == "train":
            by_source.setdefault(item["source"], []).append(item["question"])
    texts = generate([extra_question_messages(source, qs, n) for source, qs in by_source.items()])
    rows = []
    for (source, existing), text in zip(by_source.items(), texts):
        seen = {tr_lower(q) for q in existing}
        for question in parse_questions(text, n):
            if tr_lower(question) not in seen:
                seen.add(tr_lower(question))
                rows.append({"source": source, "split": "train", "question": question})
    return rows


def is_clean(row: dict | None) -> bool:
    """The judge found the answer complete and every claim supported."""
    verdict = row and row["verdict"]
    return bool(verdict) and verdict["complete"] and all(c["verdict"] == "supported" for c in verdict["claims"])


def repeats(text: str) -> bool:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]
    return len(sentences) != len(set(sentences))


def style_ok(content: str) -> bool:
    """No article codes in the answer text and no sentence said twice: two habits the first student picked up."""
    answer = json.loads(content)["answer"]
    return not BRACKET_CODE.search(answer) and not repeats(answer)


def usable(example: dict, text: str) -> bool:
    """finetune_data.parse_answer's checks for a candidate: valid JSON, no foreign script, only given articles cited."""
    answer = jd.parse(text)
    if answer is None or not answer.answer.strip() or FOREIGN_SCRIPT.search(answer.answer):
        return False
    return {(c.law_number, c.article_number) for c in answer.citations} <= given_articles(example["messages"][:-1])


def make_candidates(train: list[dict], verdicts: list[dict], generate: Generate, n: int = CANDIDATES) -> list[dict]:
    """`n` fresh answers for every training example whose answer was flagged. Rows: {"i": train index, "output"}."""
    judged = {row["i"]: row for row in verdicts}
    flagged = [i for i in range(len(train)) if not is_clean(judged.get(i))]
    index = [i for i in flagged for _ in range(n)]
    texts = generate([train[i]["messages"][:-1] for i in index])  # each prompt is sampled on its own, so answers differ
    # An unusable candidate is kept as "", so there is still one row per attempt and the judge skips it.
    return [{"i": i, "output": text if usable(train[i], text) else ""} for i, text in zip(index, texts)]


def last_message(examples: list[dict]) -> list[dict]:
    return [{"output": e["messages"][-1]["content"]} for e in examples]


def judge_file(examples: list[dict], outputs: list[dict], path: Path, judge: jd.Judge, judge_name: str) -> list[dict]:
    """Judges `outputs` against `examples`, reusing and updating the verdicts saved at `path`."""
    cached = read_jsonl(path) if path.exists() else []
    return jd.judge_outputs(examples, outputs, cached, judge, judge_name, lambda rows: write_jsonl(path, rows), len(examples))


def build(
    train: list[dict],
    train_verdicts: list[dict],
    candidates: list[dict],
    candidate_verdicts: list[dict],
    extra: list[dict],
    extra_verdicts: list[dict],
    fits: Callable[[dict], bool] = lambda example: True,
) -> tuple[list[dict], Counter]:
    """The verified training set, and how many examples each rule kept or dropped."""
    stats: Counter = Counter()
    rows = []

    def take(examples: list[dict], verdicts: list[dict], name: str) -> None:
        judged = {row["i"]: row for row in verdicts}
        for i, example in enumerate(examples):
            if not is_clean(judged.get(i)):
                stats[f"{name}: flagged"] += 1
            elif not style_ok(example["messages"][-1]["content"]):
                stats[f"{name}: style"] += 1
            else:
                stats[f"{name}: kept"] += 1
                rows.append(example)

    take(train, train_verdicts, "original")
    judged = {row["i"]: row for row in candidate_verdicts}
    repaired: set[int] = set()
    for k, candidate in enumerate(candidates):
        row = judged.get(k)
        # The verdict must be for this very text; the first clean candidate of an example wins.
        if candidate["i"] in repaired or not is_clean(row) or row["output"] != candidate["output"]:
            continue
        if not style_ok(candidate["output"]):
            continue
        repaired.add(candidate["i"])
        example = train[candidate["i"]]
        answer = {"role": "assistant", "content": candidate["output"]}
        rows.append({"messages": [*example["messages"][:-1], answer], "source": example["source"]})
    stats["repaired"] = len(repaired)
    stats["not repaired"] = len({c["i"] for c in candidates}) - len(repaired)
    take(extra, extra_verdicts, "new")

    kept = [row for row in rows if fits(row)]
    stats["too long"] = len(rows) - len(kept)
    return kept, stats


def token_fit() -> Callable[[dict], bool]:
    """True for examples the trainer will not truncate (finetune.MAX_LENGTH)."""
    from transformers import AutoTokenizer

    from .finetune import MAX_LENGTH, STUDENTS

    tokenizer = AutoTokenizer.from_pretrained(STUDENTS["4b"])

    def fits(example: dict) -> bool:
        *prompt, answer = example["messages"]
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return len(tokenizer(text + answer["content"] + tokenizer.eos_token)["input_ids"]) <= MAX_LENGTH

    return fits


def main() -> None:
    from .finetune_data import vllm_generate
    from .models import LegalAnswer

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["questions", "retrieve", "answers", "judge", "repair", "build"])
    stage = parser.parse_args().stage
    verdicts = WORK_DIR / "verdicts"
    schema = LegalAnswer.model_json_schema()

    if stage == "questions":
        rows = make_extra_questions(read_jsonl(CONTEXTS_FILE), vllm_generate(thinking=False))
        write_jsonl(V2_DIR / "questions.jsonl", rows)
        print(f"{len(rows)} new questions -> {V2_DIR / 'questions.jsonl'}")
    elif stage == "retrieve":
        rows = add_contexts(read_jsonl(V2_DIR / "questions.jsonl"), dense_retrieve)
        write_jsonl(V2_DIR / "contexts.jsonl", rows)
        print(f"{len(rows)} questions with context -> {V2_DIR / 'contexts.jsonl'}")
    elif stage == "answers":
        items = read_jsonl(V2_DIR / "contexts.jsonl")
        rows = make_answers(items, vllm_generate(thinking=True, json_schema=schema))["train"]
        write_jsonl(WORK_DIR / "extra.jsonl", rows)
        print(f"Kept {len(rows)}/{len(items)} new answers -> {WORK_DIR / 'extra.jsonl'}")
    elif stage == "judge":
        train, extra = read_jsonl(OUT_DIR / "train.jsonl"), read_jsonl(WORK_DIR / "extra.jsonl")
        jobs = {"train": (train, last_message(train)), "extra": (extra, last_message(extra))}
        if (WORK_DIR / "candidates.jsonl").exists():
            candidates = read_jsonl(WORK_DIR / "candidates.jsonl")
            jobs["candidates"] = ([train[c["i"]] for c in candidates], candidates)
        judge = jd.local_judge()
        for name, (examples, outputs) in jobs.items():
            rows = judge_file(examples, outputs, verdicts / f"{name}.jsonl", judge, jd.LOCAL_JUDGE)
            print(f"{name}: {sum(is_clean(row) for row in rows)}/{len(examples)} clean", flush=True)
    elif stage == "repair":
        train = read_jsonl(OUT_DIR / "train.jsonl")
        generate = vllm_generate(thinking=True, json_schema=schema)
        rows = make_candidates(train, read_jsonl(verdicts / "train.jsonl"), generate)
        write_jsonl(WORK_DIR / "candidates.jsonl", rows)
        print(f"{len(rows)} candidates -> {WORK_DIR / 'candidates.jsonl'}; run the judge stage again")
    else:
        train = read_jsonl(OUT_DIR / "train.jsonl")
        rows, stats = build(
            train,
            read_jsonl(verdicts / "train.jsonl"),
            read_jsonl(WORK_DIR / "candidates.jsonl"),
            read_jsonl(verdicts / "candidates.jsonl"),
            read_jsonl(WORK_DIR / "extra.jsonl"),
            read_jsonl(verdicts / "extra.jsonl"),
            token_fit(),
        )
        write_jsonl(V2_DIR / "train.jsonl", rows)
        print("\n".join(f"{key}: {count}" for key, count in stats.items()))
        print(f"{len(rows)} examples -> {V2_DIR / 'train.jsonl'}")


if __name__ == "__main__":
    main()
