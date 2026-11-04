from collections import Counter

from turkish_legal_assistant.data import ARTICLES


def test_all_laws_ingested():
    counts = Counter(a.law_number for a in ARTICLES)
    assert set(counts) == {"4857", "6698", "6502", "6098"}
    assert all(n > 30 for n in counts.values())


def test_articles_are_unique_and_non_empty():
    keys = [(a.law_number, a.article_number) for a in ARTICLES]
    assert len(keys) == len(set(keys))
    assert all(a.text for a in ARTICLES)


def test_tbk_only_contains_lease_section():
    numbers = [int(a.article_number) for a in ARTICLES if a.law_number == "6098"]
    assert numbers == list(range(299, 379))
