import json

from turkish_legal_assistant import compare as cmp


def example(reference_citations, source="4857/53"):
    return {
        "source": source,
        "messages": [
            {"role": "system", "content": "Sistem."},
            {
                "role": "user",
                "content": "Maddeler:\n[4857/53] İş Kanunu, madde 53: Yıllık ücretli izin\nMetin.\n\n"
                "[4857/54] İş Kanunu, madde 54: İzin süresinin hesabı\nMetin.\n\nSoru: Kaç gün iznim var?",
            },
            {
                "role": "assistant",
                "content": json.dumps({"answer": "14 gün.", "citations": reference_citations}),
            },
        ],
    }


def reply(citations, answer="İş Kanunu madde 53'e göre 14 gün."):
    return json.dumps({"answer": answer, "citations": citations}, ensure_ascii=False)


C53 = {"law_number": "4857", "article_number": "53"}
C54 = {"law_number": "4857", "article_number": "54"}
C99 = {"law_number": "4857", "article_number": "99"}


def test_given_articles_reads_the_refs_in_the_prompt():
    assert cmp.given_articles(example([C53])["messages"][:-1]) == {("4857", "53"), ("4857", "54")}


def test_score_good_answer():
    assert cmp.score(example([C53]), reply([C53])) == {
        "valid": True, "grounded": True, "source": True, "agree": 1.0, "foreign": False
    }


def test_score_invalid_json():
    row = cmp.score(example([C53]), "İzniniz 14 gündür.")
    assert row == {"valid": False, "grounded": False, "source": None, "agree": 0.0, "foreign": False}


def test_score_accepts_a_code_fence():
    assert cmp.score(example([C53]), f"```json\n{reply([C53])}\n```")["valid"]


def test_score_citation_outside_the_prompt_is_not_grounded():
    row = cmp.score(example([C53]), reply([C53, C99]))
    assert row["valid"] and not row["grounded"]
    assert row["agree"] == 2 * 1 / 3


def test_score_source_is_none_when_it_was_not_retrieved():
    assert cmp.score(example([C54], source="4857/1"), reply([C54]))["source"] is None


def test_score_flags_foreign_script():
    assert cmp.score(example([C53]), reply([C53], answer="İzin相关规定 14 gün."))["foreign"]


def test_f1_both_empty_is_agreement():
    assert cmp.f1(set(), set()) == 1.0
    assert cmp.f1({1}, set()) == 0.0


def test_summarize_averages_source_only_where_it_applies():
    rows = [
        {"valid": True, "grounded": True, "source": True, "agree": 1.0, "foreign": False},
        {"valid": True, "grounded": False, "source": None, "agree": 0.5, "foreign": False},
        {"valid": False, "grounded": False, "source": None, "agree": 0.0, "foreign": False},
    ]
    s = cmp.summarize(rows)
    assert s["n"] == 3 and s["source"] == 1.0 and s["agree"] == 0.5
    assert abs(s["valid"] - 2 / 3) < 1e-9


def test_run_and_report_with_a_fake_model():
    examples = [example([C53]), example([C54], source="4857/54")]
    outputs, seconds = cmp.run(examples, lambda batch: [reply([C53]) for _ in batch])
    assert [o["source"] for o in outputs] == ["4857/53", "4857/54"]
    table = cmp.report(examples, {"student": {"seconds": 1.0, "outputs": outputs}})
    assert table.splitlines()[-1].split()[:5] == ["student", "100%", "100%", "50%", "0.50"]
