"""Quality gate: the served model's committed eval results must not fall below the recorded scores.

The outputs (data/finetune/compare/) and verdicts (data/finetune/judge/) are committed, so this runs in CI with no
GPU or API. When a new adapter is trained, regenerate student-4b with compare.py and judge it again with
`judge student-4b --judge gemini`; a verdict that no longer matches its output counts as a failure here.
"""

import pytest

from turkish_legal_assistant import compare as cmp
from turkish_legal_assistant import judge as jd
from turkish_legal_assistant.finetune_data import OUT_DIR, read_jsonl

SERVED = "student-4b"
# A few points under the scores recorded on 2026-11-10, so rounding and one-question changes do not fail CI.
CITATION_FLOORS = {"valid": 0.98, "grounded": 0.95, "source": 0.70, "agree": 0.65}
JUDGE_FLOORS = {"correct": 0.38, "complete": 0.42, "faithful": 0.52}
CONTRA_CEILING = 0.40


@pytest.fixture(scope="module")
def examples():
    return read_jsonl(OUT_DIR / "eval.jsonl")


def test_citation_scores(examples):
    _, *outputs = read_jsonl(cmp.result_path(SERVED))
    s = cmp.summarize([cmp.score(e, o["output"]) for e, o in zip(examples, outputs)])
    for key, floor in CITATION_FLOORS.items():
        assert s[key] >= floor, f"{key} {s[key]:.2f} < {floor}"


def test_judge_scores(examples):
    s = jd.load_summary(SERVED, len(examples), jd.judge_dir(jd.JUDGES["gemini"]))
    assert s["judged"] == len(examples), "outputs changed or were not judged: run judge student-4b --judge gemini"
    for key, floor in JUDGE_FLOORS.items():
        assert s[key] >= floor, f"{key} {s[key]:.2f} < {floor}"
    assert s["contra"] <= CONTRA_CEILING, f"contra {s['contra']:.2f} > {CONTRA_CEILING}"
