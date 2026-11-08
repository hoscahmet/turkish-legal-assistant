"""Compare models on the answering step of RAG, using the eval split of data/finetune/ (articles unseen in training).

Every model gets the exact prompt the student was trained on (question + retrieved articles) and must reply
with LegalAnswer JSON. One model per process, because two vLLM models do not fit on the 12 GB GPU:
  uv run python -m turkish_legal_assistant.compare base|student|teacher|gemini
  uv run python -m turkish_legal_assistant.compare report
Each run writes data/finetune/compare/<model>.jsonl; `report` scores every file found there.

Scores need no judge model (that is Phase 7):
  valid:    the reply parses as LegalAnswer JSON
  grounded: valid, and every citation is one of the articles in the prompt
  source:   cites the article the question was written from, counted only when that article was retrieved
  agree:    citation F1 against the teacher's reference answer
  foreign:  the answer drifts into Chinese/Japanese/Korean script
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from pydantic import ValidationError

from .finetune_data import FOREIGN_SCRIPT, OUT_DIR, Generate, Messages, read_jsonl, write_jsonl
from .models import LegalAnswer

COMPARE_DIR = OUT_DIR / "compare"
MODELS = ["base", "student", "teacher", "gemini"]
# The articles in a prompt start lines like "[4857/53] İş Kanunu, madde 53: ..."
ARTICLE_REF = re.compile(r"^\[(\d+)/([^\]]+)\] ", re.MULTILINE)
CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
GEMINI_PAUSE = 5.0  # seconds between requests, to stay under the free tier's per-minute limit


def given_articles(prompt: Messages) -> set[tuple[str, str]]:
    return set(ARTICLE_REF.findall(prompt[-1]["content"]))


def parse(text: str) -> LegalAnswer | None:
    try:
        # Some models wrap JSON in a ```json fence even when told not to.
        return LegalAnswer.model_validate_json(CODE_FENCE.sub("", text.strip()))
    except ValidationError:
        return None


def cited(answer: LegalAnswer) -> set[tuple[str, str]]:
    return {(c.law_number, c.article_number) for c in answer.citations}


def f1(predicted: set, reference: set) -> float:
    if not predicted and not reference:
        return 1.0  # both say the articles do not answer the question
    overlap = len(predicted & reference)
    return 2 * overlap / (len(predicted) + len(reference))


def score(example: dict, output: str) -> dict:
    *prompt, reference = example["messages"]
    given = given_articles(prompt)
    source = tuple(example["source"].split("/", 1))
    answer = parse(output)
    row = {"valid": answer is not None, "grounded": False, "source": None, "agree": 0.0, "foreign": False}
    if answer is None:
        return row
    citations = cited(answer)
    row["grounded"] = citations <= given
    row["source"] = source in citations if source in given else None
    row["agree"] = f1(citations, cited(LegalAnswer.model_validate_json(reference["content"])))
    row["foreign"] = bool(FOREIGN_SCRIPT.search(answer.answer))
    return row


def summarize(rows: list[dict]) -> dict:
    """Averages over examples; `source` only over the examples where it applies."""
    n = len(rows)
    with_source = [r["source"] for r in rows if r["source"] is not None]
    return {
        "n": n,
        "valid": sum(r["valid"] for r in rows) / n,
        "grounded": sum(r["grounded"] for r in rows) / n,
        "source": sum(with_source) / len(with_source) if with_source else 0.0,
        "agree": sum(r["agree"] for r in rows) / n,
        "foreign": sum(r["foreign"] for r in rows) / n,
    }


def run(examples: list[dict], generate: Generate) -> tuple[list[dict], float]:
    """Answers every example in one batch; returns the outputs and the wall-clock seconds."""
    prompts = [example["messages"][:-1] for example in examples]
    start = time.perf_counter()
    outputs = generate(prompts)
    seconds = time.perf_counter() - start
    return [{"source": e["source"], "output": o} for e, o in zip(examples, outputs)], seconds


def report(examples: list[dict], results: dict[str, dict]) -> str:
    header = f"{'model':<8} {'valid':>6} {'grounded':>9} {'source':>7} {'agree':>6} {'foreign':>8} {'sec/q':>6}"
    lines = [header, "-" * len(header)]
    for name, result in results.items():
        s = summarize([score(e, o["output"]) for e, o in zip(examples, result["outputs"])])
        lines.append(
            f"{name:<8} {s['valid']:>6.0%} {s['grounded']:>9.0%} {s['source']:>7.0%} {s['agree']:>6.2f}"
            f" {s['foreign']:>8.0%} {result['seconds'] / s['n']:>6.2f}"
        )
    return "\n".join(lines)


def gemini_generate() -> Generate:
    """Plain text replies (no structured output), so Gemini is judged on the same instructions as the others."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from .agent import get_model

    os.environ["LLM_PROVIDER"] = "gemini"
    model = get_model()

    def generate(batch: list[Messages]) -> list[str]:
        texts = []
        for system, user in batch:
            for attempt in range(3):
                try:
                    texts.append(model.invoke([SystemMessage(system["content"]), HumanMessage(user["content"])]).text)
                    break
                except Exception as e:  # free tier: rate limits and overloads are temporary
                    print(f"Gemini error ({e}); retrying")
                    time.sleep(30 * (attempt + 1))
            else:
                texts.append("")
            time.sleep(GEMINI_PAUSE)
        return texts

    return generate


def local_generate(name: str) -> Generate:
    from .finetune import ADAPTER_DIR, STUDENT
    from .finetune_data import vllm_generate

    if name == "teacher":
        # As when it wrote the reference answers: thinking on, JSON enforced.
        return vllm_generate(thinking=True, json_schema=LegalAnswer.model_json_schema())
    # The student was trained with thinking off and no JSON constraint; the base model gets the same setup.
    return vllm_generate(thinking=False, model=STUDENT, adapter=ADAPTER_DIR if name == "student" else None)


def result_path(name: str, out_dir: Path = COMPARE_DIR) -> Path:
    return out_dir / f"{name}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=[*MODELS, "report"])
    args = parser.parse_args()
    examples = read_jsonl(OUT_DIR / "eval.jsonl")

    if args.model == "report":
        results = {}
        for name in MODELS:
            if result_path(name).exists():
                meta, *outputs = read_jsonl(result_path(name))
                results[name] = {"seconds": meta["seconds"], "outputs": outputs}
        print(report(examples, results) if results else f"No results in {COMPARE_DIR}")
        return

    generate = gemini_generate() if args.model == "gemini" else local_generate(args.model)
    outputs, seconds = run(examples, generate)
    # First line holds the timing, the rest one output per eval example in order.
    write_jsonl(result_path(args.model), [{"seconds": seconds}, *outputs])
    print(f"{len(outputs)} answers in {seconds:.0f} s -> {result_path(args.model)}")


if __name__ == "__main__":
    main()
