import json

import pytest

from turkish_legal_assistant import finetune_data as fd
from turkish_legal_assistant.data import ARTICLES, find_article

IZIN = find_article("4857", "53")
KIRA = find_article("6098", "342")


def cite(*articles, answer="Cevap."):
    return json.dumps(
        {"answer": answer, "citations": [{"law_number": a.law_number, "article_number": a.article_number} for a in articles]}
    )


def test_parse_questions_strips_numbering_and_junk():
    text = "İşte sorular:\n1. Kaç gün iznim var?\n- Patron izni bölebilir mi?\n3) Üçüncü soru?"
    assert fd.parse_questions(text) == ["Kaç gün iznim var?", "Patron izni bölebilir mi?"]


def test_parse_questions_drops_questions_that_point_at_the_article():
    text = "Bu Kanun ev işlerini kapsar mı?\nEvde temizlik yapan bir kadınım, iş kanunu bana uygulanır mı?"
    assert fd.parse_questions(text) == ["Evde temizlik yapan bir kadınım, iş kanunu bana uygulanır mı?"]


def test_strip_thinking_keeps_only_the_reply():
    assert fd.strip_thinking('<think>\nMadde 53 izin süresini...\n</think>\n\n{"answer": "x"}') == '{"answer": "x"}'
    assert fd.strip_thinking('{"answer": "x"}') == '{"answer": "x"}'


def test_parse_answer_rejects_bad_json_and_foreign_citations():
    assert fd.parse_answer(cite(IZIN), [IZIN]).citations[0].article_number == "53"
    assert fd.parse_answer("not json", [IZIN]) is None
    assert fd.parse_answer(cite(KIRA), [IZIN]) is None  # cites an article it was not given
    assert fd.parse_answer(cite(answer=" "), [IZIN]) is None
    assert fd.parse_answer(cite(IZIN, answer="İş Kanunu'ndaki相关规定 include: izin."), [IZIN]) is None
    # "The articles do not answer this" with no citations is a valid lesson for the student.
    assert fd.parse_answer(cite(answer="Bu maddelerde cevap yok."), [IZIN]) is not None


def test_repealed_articles_are_skipped():
    repealed = find_article("4857", "33")
    assert fd.is_repealed(repealed) and not fd.is_repealed(IZIN)


def test_build_examples():
    sources = ARTICLES[:11]  # article 0 lands in eval (every 10th), the rest in train
    calls = []

    def generate(batch):
        calls.append(batch)
        if len(calls) == 1:  # questions: two per article
            return ["Birinci soru nedir?\nİkinci soru nedir?"] * len(batch)
        # answers: every other one cites an article it was not given and is dropped
        return [cite(KIRA) if i % 2 else cite(IZIN) for i in range(len(batch))]

    splits = fd.build_examples(sources, generate, retrieve=lambda q: [IZIN])

    sources_left = [a for a in sources if not fd.is_repealed(a)]
    assert len(calls[1]) == 2 * len(sources_left)
    assert len(splits["train"]) + len(splits["eval"]) == len(sources_left)
    example = splits["eval"][0]
    assert [m["role"] for m in example["messages"]] == ["system", "user", "assistant"]
    assert "Soru: Birinci soru nedir?" in example["messages"][1]["content"]
    assert "[4857/53]" in example["messages"][1]["content"]
    assert example["source"] == f"{sources_left[0].law_number}/{sources_left[0].article_number}"


def test_save_writes_jsonl(tmp_path):
    fd.save({"train": [{"messages": [], "source": "4857/53"}], "eval": []}, tmp_path)
    assert json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])["source"] == "4857/53"


def test_stages_round_trip_through_files(tmp_path):
    """The three GPU stages run as separate processes, so everything must survive a trip through JSONL."""
    questions = fd.make_questions(ARTICLES[:5], lambda batch: ["Yıllık iznim kaç gün?"] * len(batch))
    assert questions[0] == {"source": f"{ARTICLES[0].law_number}/{ARTICLES[0].article_number}", "split": "eval",
                            "question": "Yıllık iznim kaç gün?"}
    assert {q["split"] for q in questions[1:]} == {"train"}

    fd.write_jsonl(tmp_path / "q.jsonl", questions)
    contexts = fd.add_contexts(fd.read_jsonl(tmp_path / "q.jsonl"), retrieve=lambda q: [IZIN, KIRA])
    assert contexts[0]["context"] == ["4857/53", "6098/342"]

    fd.write_jsonl(tmp_path / "c.jsonl", contexts)
    splits = fd.make_answers(fd.read_jsonl(tmp_path / "c.jsonl"), lambda batch: [cite(KIRA)] * len(batch))
    assert len(splits["eval"]) == 1 and len(splits["train"]) == len(questions) - 1
    assert "[6098/342]" in splits["train"][0]["messages"][1]["content"]


def test_from_ref_handles_article_numbers_with_spaces():
    geçici = next(a for a in ARTICLES if " " in a.article_number)  # e.g. "Geçici 1"
    assert fd.from_ref(fd.ref(geçici)) == geçici


def test_read_jsonl_explains_missing_stage(tmp_path):
    with pytest.raises(FileNotFoundError, match="previous stage"):
        fd.read_jsonl(tmp_path / "missing.jsonl")
