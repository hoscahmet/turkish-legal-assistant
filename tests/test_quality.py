"""Quality gate: the served model's committed eval results must not fall below the recorded scores.

The outputs (data/finetune/compare/) and verdicts (data/finetune/judge/) are committed, so this runs in CI with no
GPU or API. When a new adapter is trained, regenerate student-4b-v2 with compare.py and judge it again with
`judge student-4b-v2`; a verdict that no longer matches its output counts as a failure here.
"""

import pytest

from turkish_legal_assistant import compare as cmp
from turkish_legal_assistant import judge as jd
from turkish_legal_assistant.finetune_data import OUT_DIR, read_jsonl

SERVED = "student-4b-v2"
# A few points under the scores recorded on 2026-11-11, so rounding and one-question changes do not fail CI.
CITATION_FLOORS = {"valid": 0.98, "grounded": 0.95, "source": 0.70, "agree": 0.75}
# Local judge (Qwen3-8B). It is lenient, so these floors are higher than the Gemini ones.
LOCAL_FLOORS = {"correct": 0.66, "complete": 0.66, "faithful": 0.68}
LOCAL_CONTRA_CEILING = 0.17
# Gemini 3.5 Flash, the judge to trust: the served model must at least match the first 4B student's recorded level.
GEMINI_FLOORS = {"correct": 0.38, "complete": 0.42, "faithful": 0.52}
GEMINI_CONTRA_CEILING = 0.40


@pytest.fixture(scope="module")
def examples():
    return read_jsonl(OUT_DIR / "eval.jsonl")


def check(summary: dict, floors: dict, contra_ceiling: float) -> None:
    for key, floor in floors.items():
        assert summary[key] >= floor, f"{key} {summary[key]:.2f} < {floor}"
    assert summary["contra"] <= contra_ceiling, f"contra {summary['contra']:.2f} > {contra_ceiling}"


def test_citation_scores(examples):
    _, *outputs = read_jsonl(cmp.result_path(SERVED))
    s = cmp.summarize([cmp.score(e, o["output"]) for e, o in zip(examples, outputs)])
    for key, floor in CITATION_FLOORS.items():
        assert s[key] >= floor, f"{key} {s[key]:.2f} < {floor}"


def test_local_judge_scores(examples):
    s = jd.load_summary(SERVED, len(examples), jd.judge_dir(jd.JUDGES["local"]))
    assert s["judged"] == len(examples), f"outputs changed or were not judged: run judge {SERVED}"
    check(s, LOCAL_FLOORS, LOCAL_CONTRA_CEILING)


def test_gemini_judge_scores(examples):
    s = jd.load_summary(SERVED, len(examples), jd.judge_dir(jd.JUDGES["gemini"]))
    if s["judged"] < len(examples):
        # The Gemini run needs a paid key (or three days of the free tier); until then only the local judge gates.
        pytest.skip(f"{SERVED}: {s['judged']}/{len(examples)} answers judged by Gemini; run judge {SERVED} --judge gemini")
    check(s, GEMINI_FLOORS, GEMINI_CONTRA_CEILING)
