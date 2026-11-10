import json

from turkish_legal_assistant import judge as jd
from turkish_legal_assistant.finetune_data import write_jsonl

EXAMPLE = {
    "source": "4857/53",
    "messages": [
        {"role": "system", "content": "Sistem."},
        {"role": "user", "content": "Maddeler:\n[4857/53] İş Kanunu, madde 53: Yıllık izin\n14 gün.\n\nSoru: Kaç gün?"},
        {"role": "assistant", "content": "{}"},
    ],
}
GOOD = json.dumps({"answer": "14 gün izin hakkınız var.", "citations": []}, ensure_ascii=False)


def verdict(*labels, complete=True):
    return jd.Verdict(claims=[jd.Claim(claim=f"iddia {i}", verdict=v) for i, v in enumerate(labels)],
                      complete=complete, note="")


def fake_judge(result):
    calls = []

    def judge(batch):
        calls.append(len(batch))
        return [result] * len(batch)

    return judge, calls


def test_judge_prompt_has_the_articles_question_and_answer():
    prompt = jd.judge_prompt(EXAMPLE, jd.parse(GOOD))
    assert "[4857/53]" in prompt and "Soru: Kaç gün?" in prompt and prompt.endswith("Cevap: 14 gün izin hakkınız var.")


def test_invalid_output_is_not_sent_to_the_judge():
    judge, calls = fake_judge(verdict("supported"))
    rows = jd.judge_outputs([EXAMPLE], [{"output": "not json"}], [], judge, "j")
    assert rows == [{"i": 0, "output": "not json", "verdict": None, "judge": None}] and calls == []


def test_unchanged_output_reuses_the_cached_verdict_and_changed_output_is_judged_again():
    judge, calls = fake_judge(verdict("contradicted"))
    cached = [{"i": 0, "output": GOOD, "verdict": verdict("supported").model_dump(), "judge": "judge-a"}]
    assert jd.judge_outputs([EXAMPLE], [{"output": GOOD}], cached, judge, judge_name="judge-a") == cached
    assert calls == []

    changed = GOOD.replace("14", "20")
    rows = jd.judge_outputs([EXAMPLE], [{"output": changed}], cached, judge, judge_name="judge-a")
    assert len(calls) == 1 and rows[0]["output"] == changed
    assert rows[0]["verdict"]["claims"][0]["verdict"] == "contradicted"


def test_a_different_judge_judges_again():
    judge, calls = fake_judge(verdict("supported"))
    cached = [{"i": 0, "output": GOOD, "verdict": verdict("supported").model_dump(), "judge": "judge-a"}]
    rows = jd.judge_outputs([EXAMPLE], [{"output": GOOD}], cached, judge, judge_name="judge-b")
    assert len(calls) == 1 and rows[0]["judge"] == "judge-b"


def test_failed_judgement_is_left_out_for_the_next_run():
    judge, _ = fake_judge(None)
    assert jd.judge_outputs([EXAMPLE], [{"output": GOOD}], [], judge, "j") == []


def test_answers_go_to_the_judge_in_batches_and_are_saved_after_each():
    saved = []
    judge, calls = fake_judge(verdict("supported"))
    examples, outputs = [EXAMPLE] * 3, [{"output": GOOD}] * 3
    rows = jd.judge_outputs(examples, outputs, [], judge, "j", lambda rows: saved.append(list(rows)), batch_size=2)
    assert calls == [2, 1] and [len(r) for r in saved] == [2, 3] and [r["i"] for r in rows] == [0, 1, 2]


def test_parse_verdict():
    assert jd.parse_verdict(verdict("supported").model_dump_json()).claims[0].verdict == "supported"
    assert jd.parse_verdict("") is None and jd.parse_verdict('{"claims": []}') is None


def test_score():
    row = lambda v: {"i": 0, "output": "", "verdict": v.model_dump()}  # noqa: E731
    good = jd.score(row(verdict("supported", "supported")))
    assert good == {"correct": True, "complete": True, "faithful": True, "contra": False, "claims": 2, "supported": 2}
    wrong = jd.score(row(verdict("supported", "contradicted")))
    assert not wrong["correct"] and wrong["contra"] and not wrong["faithful"]
    extra = jd.score(row(verdict("supported", "unsupported")))
    assert extra["correct"] and not extra["faithful"] and not extra["contra"]
    assert not jd.score(row(verdict("supported", complete=False)))["correct"]
    # "The articles do not answer this" has no claims: faithful by definition, correct if the judge agrees.
    assert jd.score(row(verdict()))["faithful"]
    assert jd.score(None) == jd.score({"i": 0, "output": "x", "verdict": None})
    assert not jd.score(None)["correct"]


def test_summarize_counts_missing_and_invalid_answers_as_failures():
    rows = [
        {"i": 0, "output": "", "verdict": verdict("supported", "supported").model_dump()},
        {"i": 1, "output": "", "verdict": verdict("contradicted", "supported").model_dump()},
        {"i": 2, "output": "", "verdict": None},
    ]
    s = jd.summarize(rows, n=4)
    assert s["correct"] == 0.25 and s["contra"] == 0.25 and s["judged"] == 3
    assert s["support"] == 3 / 4


def test_load_summary_ignores_verdicts_for_outputs_that_changed(tmp_path):
    compare_dir, judge_dir = tmp_path / "compare", tmp_path / "judge"
    compare_dir.mkdir(), judge_dir.mkdir()
    write_jsonl(compare_dir / "student.jsonl", [{"seconds": 1}, {"output": GOOD}, {"output": "new"}])
    write_jsonl(judge_dir / "student.jsonl", [
        {"i": 0, "output": GOOD, "verdict": verdict("supported").model_dump()},
        {"i": 1, "output": "old", "verdict": verdict("supported").model_dump()},
    ])
    s = jd.load_summary("student", 2, judge_dir, compare_dir)
    assert s["judged"] == 1 and s["correct"] == 0.5


def test_each_judge_has_its_own_folder():
    assert jd.judge_dir("Qwen/Qwen3-8B-AWQ").name == "qwen3-8b-awq"
    assert jd.judge_dir("gemini-3.5-flash").name == "gemini-3.5-flash"


def test_report_lists_every_model():
    s = jd.summarize([{"i": 0, "output": "", "verdict": verdict("supported").model_dump()}], n=1)
    text = jd.report({"student-4b": s, "gemini": s}, n=1)
    assert "student-4b" in text and "gemini" in text and "100%" in text
