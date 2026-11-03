from turkish_legal_assistant.text import tr_lower


def test_tr_lower_handles_dotted_and_dotless_i():
    assert tr_lower("IŞIK") == "ışık"
    assert tr_lower("İŞÇİ") == "işçi"
    assert tr_lower("KİRA BEDELİ") == "kira bedeli"
