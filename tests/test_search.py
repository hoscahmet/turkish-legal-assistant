from turkish_legal_assistant.search import search


def test_finds_relevant_article():
    top = search("Yıllık İZİN süresi kaç gündür?", top_k=3)[0].article
    assert (top.law_number, top.article_number) == ("4857", "53")


def test_uppercase_turkish_matches():
    # "KİRACI" only matches "kiracı" if İ/I are lowercased the Turkish way.
    assert search("KİRACI KİRAYA VEREN", top_k=1)[0].article.law_number == "6098"


def test_no_match_and_law_filter():
    assert search("uzay gemisi", top_k=3) == []
    assert all(r.article.law_number == "6698" for r in search("veri", top_k=5, law_number="6698"))
