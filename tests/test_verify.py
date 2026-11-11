import json

from turkish_legal_assistant import judge as jd
from turkish_legal_assistant import verify as v
from turkish_legal_assistant.finetune_data import OUT_DIR, read_jsonl

USER = "Maddeler:\n[4857/53] İş Kanunu, madde 53: Yıllık izin\n14 gün.\n\nSoru: Kaç gün?"


def answer(text="İş Kanunu madde 53'e göre 14 gün izin hakkınız var.", *cited):
    citations = [{"law_number": law, "article_number": article} for law, article in cited]
    return json.dumps({"answer": text, "citations": citations}, ensure_ascii=False)


def example(content=None, source="4857/53"):
    messages = [{"role": "system", "content": "Sistem."}, {"role": "user", "content": USER}]
    return {"messages": [*messages, {"role": "assistant", "content": content or answer()}], "source": source}


def row(i, output, *labels, complete=True):
    claims = [{"claim": f"iddia {k}", "verdict": label} for k, label in enumerate(labels)]
    return {"i": i, "output": output, "verdict": {"claims": claims, "complete": complete, "note": ""}, "judge": "j"}


def test_is_clean_needs_a_complete_answer_with_every_claim_supported():
    assert v.is_clean(row(0, "x", "supported", "supported"))
    assert not v.is_clean(row(0, "x", "supported", "unsupported"))
    assert not v.is_clean(row(0, "x", "contradicted"))
    assert not v.is_clean(row(0, "x", "supported", complete=False))
    assert not v.is_clean({"i": 0, "output": "x", "verdict": None, "judge": None})  # not valid JSON, never judged
    assert not v.is_clean(None)  # no verdict yet


def test_style_ok_rejects_article_codes_and_repeated_sentences():
    assert v.style_ok(answer())
    assert not v.style_ok(answer("Bu durum [4857/53] maddesinde düzenlenmiştir."))
    assert not v.style_ok(answer("İş Kanunu 4857/53'e göre 14 gün izin hakkınız var."))
    twice = "İşveren bu durumda ücreti ödemek zorundadır."
    assert not v.style_ok(answer(f"{twice} Ayrıca faiz de işler. {twice}"))


def test_usable_needs_valid_json_citing_only_given_articles():
    assert v.usable(example(), answer("14 gün.", ("4857", "53")))
    assert not v.usable(example(), answer("14 gün.", ("4857", "54")))  # not in the prompt
    assert not v.usable(example(), "14 gün.")
    assert not v.usable(example(), answer(" "))


def test_extra_questions_are_new_and_only_for_training_articles():
    items = [
        {"source": "4857/53", "split": "train", "question": "Kaç gün iznim var?"},
        {"source": "4857/54", "split": "eval", "question": "İzne ne zaman hak kazanırım?"},
    ]
    prompts = []

    def generate(batch):
        prompts.extend(batch)
        return ["Kaç gün iznim var?\nYaşım elliyi geçti, iznim artar mı?\nYeraltında çalışıyorum, iznim farklı mı?"]

    rows = v.make_extra_questions(items, generate)
    assert len(prompts) == 1 and "- Kaç gün iznim var?" in prompts[0][0]["content"]  # the eval article is never asked about
    assert [r["question"] for r in rows] == ["Yaşım elliyi geçti, iznim artar mı?", "Yeraltında çalışıyorum, iznim farklı mı?"]
    assert all(r == {"source": "4857/53", "split": "train", "question": r["question"]} for r in rows)


def test_candidates_are_written_only_for_flagged_answers():
    train = [example(), example(), example()]
    verdicts = [row(0, "x", "supported"), row(1, "x", "contradicted")]  # 2 was never judged
    asked = []

    def generate(batch):
        asked.extend(batch)
        return [answer("14 gün.", ("4857", "53")), "bozuk"] * (len(batch) // 2)

    rows = v.make_candidates(train, verdicts, generate, n=2)
    assert [r["i"] for r in rows] == [1, 1, 2, 2] and len(asked) == 4
    assert all(m[-1]["role"] == "user" for m in asked)  # the prompt only, without the flagged answer
    assert [bool(r["output"]) for r in rows] == [True, False, True, False]  # unusable candidates become ""


def test_build_keeps_clean_answers_and_the_first_clean_candidate():
    bracket = answer("Bu konu [4857/53] maddesinde düzenlenir.")
    train = [example(), example(answer("Yanlış.")), example(bracket), example(answer("Eksik."))]
    train_verdicts = [row(0, "a", "supported"), row(1, "b", "contradicted"), row(2, "c", "supported"),
                      row(3, "d", "supported", complete=False)]
    good, better = answer("14 gün izin hakkınız var."), answer("En az 14 gün izin hakkınız var.")
    candidates = [{"i": 1, "output": ""}, {"i": 1, "output": good}, {"i": 1, "output": better}, {"i": 3, "output": good}]
    candidate_verdicts = [
        {"i": 0, "output": "", "verdict": None, "judge": None},
        row(1, good, "supported"),
        row(2, better, "supported"),
        row(3, "an older text", "supported"),  # a verdict for a different output does not count
    ]
    extra = [example(source="4857/54"), example(answer("Uydurma."), source="4857/55")]
    extra_verdicts = [row(0, "e", "supported"), row(1, "f", "unsupported")]

    rows, stats = v.build(train, train_verdicts, candidates, candidate_verdicts, extra, extra_verdicts)
    contents = [r["messages"][-1]["content"] for r in rows]
    assert contents == [answer(), good, answer()]  # clean original, example 1 repaired by its first clean candidate, clean new
    assert rows[1]["messages"][:-1] == train[1]["messages"][:-1] and rows[1]["source"] == "4857/53"
    assert stats == {"original: kept": 1, "original: flagged": 2, "original: style": 1, "repaired": 1, "not repaired": 1,
                     "new: kept": 1, "new: flagged": 1, "too long": 0}

    rows, stats = v.build(train, train_verdicts, candidates, candidate_verdicts, extra, extra_verdicts, fits=lambda e: False)
    assert rows == [] and stats["too long"] == 3


def test_judge_file_saves_and_reuses_verdicts(tmp_path):
    path = tmp_path / "verdicts" / "train.jsonl"
    examples = [example(), example()]
    calls = []

    def judge(batch):
        calls.append(len(batch))
        return [jd.Verdict(claims=[jd.Claim(claim="iddia", verdict="supported")], complete=True, note="")] * len(batch)

    rows = v.judge_file(examples, v.last_message(examples), path, judge, "j")
    assert calls == [2] and all(v.is_clean(r) for r in rows) and read_jsonl(path) == rows
    v.judge_file(examples, v.last_message(examples), path, judge, "j")
    assert calls == [2]  # nothing changed, nothing judged again


def test_committed_verified_set_keeps_the_eval_split_unseen():
    train = read_jsonl(v.V2_DIR / "train.jsonl")
    evaluation = read_jsonl(OUT_DIR / "eval.jsonl")
    eval_questions = {e["messages"][1]["content"].rsplit("Soru: ", 1)[1] for e in evaluation}
    assert not {e["source"] for e in train} & {e["source"] for e in evaluation}
    assert not {e["messages"][1]["content"].rsplit("Soru: ", 1)[1] for e in train} & eval_questions
    assert all(v.style_ok(e["messages"][-1]["content"]) and v.usable(e, e["messages"][-1]["content"]) for e in train)
    assert all(q["split"] == "train" for q in read_jsonl(v.V2_DIR / "questions.jsonl"))
