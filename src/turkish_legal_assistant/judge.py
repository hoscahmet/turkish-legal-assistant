"""LLM-as-a-judge: is what each answer says about the law actually in the articles it was given?

  uv run python -m turkish_legal_assistant.judge [model ...] [--judge local|gemini]   # default: every result, local
  uv run python -m turkish_legal_assistant.judge report [--judge local|gemini]

compare.py only checks citations. Here a judge model splits each answer from data/finetune/compare/ into claims
and labels every claim against the articles in the prompt:
  supported     an article states it or it follows directly
  contradicted  an article says otherwise ("the law does not say" about something it does say counts)
  unsupported   the articles neither state nor contradict it
and says whether the answer is complete (gives the rule, number or condition the question needs).

Scores per model, over all eval questions (an answer that is not valid JSON fails every score):
  correct:  complete and no contradicted claim
  complete: the judge's completeness verdict
  faithful: every claim is supported
  contra:   the answer has at least one contradicted claim
  support:  share of all claims that are supported

Verdicts are saved per judge in data/finetune/judge/<judge>/<model>.jsonl and committed, so the
report and the CI quality gate need no API. A verdict is reused only while the output it judged and the judge
model are unchanged.
Two judges:
  local   the Phase 6 teacher, Qwen3-8B, on the GPU with thinking on: free, ~10 minutes for every model. It wrote the
          reference answers and the student's training data, so the teacher and student rows may be scored generously.
  gemini  Gemini 3.5 Flash (JUDGE_MODEL overrides it): independent of every judged model, but its free tier allows
          only 20 requests a day, so a full run needs a paid key.
"""

import argparse
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .compare import COMPARE_DIR, MODELS, parse, result_path
from .finetune_data import OUT_DIR, read_jsonl, write_jsonl
from .models import LegalAnswer

JUDGE_DIR = OUT_DIR / "judge"
# gemini-3.8-flash timed out (504) when tried.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.5-flash")
JUDGE_PAUSE = 1.0  # seconds after each request
GEMINI_PARALLEL = 8  # requests in flight; the paid tier allows 1,000 a minute

JUDGE_SYSTEM = """You check answers written by a Turkish legal information assistant. You get the law articles
the assistant was given, the user's question and the assistant's answer.
1. Split the answer into its factual claims about the law. Skip disclaimers ("hukuki tavsiye değildir"),
   advice to see a lawyer, and restatements of the question.
2. Label each claim using only the given articles, not your own knowledge of Turkish law:
   - supported: an article states it, or it follows directly from an article.
   - contradicted: an article says otherwise. Claiming the articles do not cover something they do cover
     counts as contradicted.
   - unsupported: the articles neither state nor contradict it.
3. complete: true if the answer gives what the articles say that the question needs (the key rule, number
   or condition); false if it leaves that out or hedges on it. If the articles do not answer the question,
   an answer that says so is complete.
Write the claims and the note in Turkish. The note is one short sentence on the main problem, or empty."""


class Claim(BaseModel):
    claim: str
    verdict: Literal["supported", "contradicted", "unsupported"]


class Verdict(BaseModel):
    claims: list[Claim] = Field(description="Factual claims about the law in the answer, each with its label.")
    complete: bool = Field(description="The answer gives what the articles say that the question needs.")
    note: str = Field(description="One short sentence in Turkish on the main problem, or empty.")


# Takes (system, user) message pairs, returns one verdict per pair (None where the judge failed).
Judge = Callable[[list[tuple[str, str]]], list[Verdict | None]]
LOCAL_JUDGE = "Qwen/Qwen3-8B-AWQ"
JUDGES = {"local": LOCAL_JUDGE, "gemini": JUDGE_MODEL}


def judge_dir(judge_name: str) -> Path:
    """Each judge keeps its own verdicts, so two judges can be compared: judge/qwen3-8b-awq/, judge/gemini-3.5-flash/."""
    return JUDGE_DIR / judge_name.rsplit("/", 1)[-1].lower()


def judge_prompt(example: dict, answer: LegalAnswer) -> str:
    """The student's own user message (articles + question), then the answer to check."""
    context = example["messages"][1]["content"]
    return f"{context}\n\nCevap: {answer.answer}"


def judge_outputs(examples: list[dict], outputs: list[dict], cached: list[dict], judge: Judge, judge_name: str,
                  save: Callable[[list[dict]], None] = lambda rows: None, batch_size: int = 1) -> list[dict]:
    """One row per example: the output judged and its verdict (None when the output is not valid JSON).

    Rows in `cached` with the same output and judge are reused. Answers go to the judge `batch_size` at a time and
    `save` gets the rows so far after each batch, so a crash or rate-limit stop loses little; failed judgements are
    left out and retried on the next run.
    """
    previous = {row["i"]: row for row in cached}
    rows, pending = {}, []
    for i, (example, output) in enumerate(zip(examples, outputs)):
        text = output["output"]
        old = previous.get(i)
        if old and old["output"] == text and (old["verdict"] is None or old.get("judge") == judge_name):
            rows[i] = old
        elif (answer := parse(text)) is None:
            rows[i] = {"i": i, "output": text, "verdict": None, "judge": None}
        else:
            pending.append((i, text, (JUDGE_SYSTEM, judge_prompt(example, answer))))
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        for (i, text, _), verdict in zip(batch, judge([prompt for _, _, prompt in batch])):
            if verdict is not None:
                rows[i] = {"i": i, "output": text, "verdict": verdict.model_dump(), "judge": judge_name}
        save([rows[i] for i in sorted(rows)])
    return [rows[i] for i in sorted(rows)]


def score(row: dict | None) -> dict:
    verdict = row and row["verdict"]
    if not verdict:
        return {"correct": False, "complete": False, "faithful": False, "contra": False, "claims": 0, "supported": 0}
    labels = [c["verdict"] for c in verdict["claims"]]
    contra = "contradicted" in labels
    return {
        "correct": verdict["complete"] and not contra,
        "complete": verdict["complete"],
        "faithful": all(label == "supported" for label in labels),
        "contra": contra,
        "claims": len(labels),
        "supported": labels.count("supported"),
    }


def summarize(rows: list[dict], n: int) -> dict:
    """Rates over all n eval questions; missing rows (not judged yet) count as failures."""
    by_index = {row["i"]: row for row in rows}
    scores = [score(by_index.get(i)) for i in range(n)]
    claims = sum(s["claims"] for s in scores)
    summary = {key: sum(s[key] for s in scores) / n for key in ("correct", "complete", "faithful", "contra")}
    summary["support"] = sum(s["supported"] for s in scores) / claims if claims else 0.0
    summary["judged"] = len(rows)
    return summary


def report(results: dict[str, dict], n: int) -> str:
    header = f"{'model':<14} {'judged':>7} {'correct':>8} {'complete':>9} {'faithful':>9} {'contra':>7} {'support':>8}"
    lines = [header, "-" * len(header)]
    for name, s in results.items():
        lines.append(
            f"{name:<14} {s['judged']:>4}/{n:<2} {s['correct']:>8.0%} {s['complete']:>9.0%} {s['faithful']:>9.0%}"
            f" {s['contra']:>7.0%} {s['support']:>8.0%}"
        )
    return "\n".join(lines)


def judge_path(name: str, out_dir: Path = JUDGE_DIR) -> Path:
    return out_dir / f"{name}.jsonl"


def load_judged(name: str, out_dir: Path = JUDGE_DIR) -> list[dict]:
    path = judge_path(name, out_dir)
    return read_jsonl(path) if path.exists() else []


def load_summary(name: str, n: int, judged_dir: Path, compare_dir: Path = COMPARE_DIR) -> dict:
    """Scores the saved verdicts that still match the saved outputs of `name`."""
    _, *outputs = read_jsonl(result_path(name, compare_dir))
    current = {i: o["output"] for i, o in enumerate(outputs)}
    rows = [row for row in load_judged(name, judged_dir) if current.get(row["i"]) == row["output"]]
    return summarize(rows, n)


def gemini_judge() -> Judge:
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage, SystemMessage

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise SystemExit("Set GEMINI_API_KEY to run the judge")
    model = init_chat_model(JUDGE_MODEL, model_provider="google_genai", temperature=0).with_structured_output(Verdict)

    def one(system: str, user: str) -> Verdict | None:
        try:
            for attempt in range(3):
                try:
                    return model.invoke([SystemMessage(system), HumanMessage(user)])
                except Exception as e:  # rate limits and overloads are usually temporary
                    print(f"Judge error ({e}); retrying", flush=True)
                    time.sleep(30 * (attempt + 1))
            return None
        finally:
            time.sleep(JUDGE_PAUSE)

    def judge(batch: list[tuple[str, str]]) -> list[Verdict | None]:
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            return list(pool.map(lambda pair: one(*pair), batch))

    return judge


def parse_verdict(text: str) -> Verdict | None:
    try:
        return Verdict.model_validate_json(text)
    except ValidationError:
        return None


def local_judge() -> Judge:
    from .finetune_data import vllm_generate

    # Thinking on and the JSON schema enforced after it, as when this model wrote the reference answers.
    generate = vllm_generate(thinking=True, json_schema=Verdict.model_json_schema(), model=LOCAL_JUDGE)

    def judge(batch: list[tuple[str, str]]) -> list[Verdict | None]:
        texts = generate([[{"role": "system", "content": s}, {"role": "user", "content": u}] for s, u in batch])
        return [parse_verdict(text) for text in texts]

    return judge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", choices=[*MODELS, "report"], help="default: every compare result")
    parser.add_argument("--judge", choices=["local", "gemini"], default="local")
    args = parser.parse_args()
    examples = read_jsonl(OUT_DIR / "eval.jsonl")
    available = [name for name in MODELS if result_path(name).exists()]

    judge_name = JUDGES[args.judge]
    out_dir = judge_dir(judge_name)
    print(f"Judge: {judge_name}", flush=True)

    if args.models == ["report"]:
        results = {name: load_summary(name, len(examples), out_dir) for name in available}
        print(report(results, len(examples)) if results else f"No results in {COMPARE_DIR}")
        return

    if args.judge == "local":
        judge, batch_size = local_judge(), len(examples)
    else:
        judge, batch_size = gemini_judge(), GEMINI_PARALLEL
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.models or available:
        _, *outputs = read_jsonl(result_path(name))

        def save(rows: list[dict], name: str = name) -> None:
            write_jsonl(judge_path(name, out_dir), rows)

        rows = judge_outputs(examples, outputs, load_judged(name, out_dir), judge, judge_name, save, batch_size)
        print(f"{name}: {len(rows)}/{len(examples)} judged -> {judge_path(name, out_dir)}", flush=True)


if __name__ == "__main__":
    main()
