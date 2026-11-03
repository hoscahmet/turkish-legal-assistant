from fastapi.testclient import TestClient

from turkish_legal_assistant.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_laws():
    numbers = {law["number"] for law in client.get("/laws").json()}
    assert numbers == {"4857", "6698", "6502", "6098"}


def test_get_article():
    r = client.get("/laws/4857/articles/53")
    assert r.status_code == 200
    assert r.json()["title"] == "Yıllık ücretli izin hakkı ve izin süreleri"


def test_get_article_not_found():
    assert client.get("/laws/4857/articles/9999").status_code == 404


def test_ask_finds_relevant_article():
    r = client.post("/ask", json={"question": "Yıllık İZİN süresi kaç gündür?"})
    assert r.status_code == 200
    body = r.json()
    top = body["sources"][0]["article"]
    assert (top["law_number"], top["article_number"]) == ("4857", "53")
    assert "İş Kanunu madde 53" in body["answer"]


def test_ask_uppercase_turkish_matches():
    # "KİRA" only matches "kira" if İ/I are lowercased the Turkish way.
    r = client.post("/ask", json={"question": "KİRACI KİRA BEDELİNİ ÖDEMEZSE"})
    assert r.json()["sources"][0]["article"]["law_number"] == "6098"


def test_ask_no_match():
    r = client.post("/ask", json={"question": "uzay gemisi"})
    assert r.json()["sources"] == []
    assert r.json()["answer"] == "Sorunuzla ilgili bir madde bulunamadı."


def test_ask_validates_input():
    assert client.post("/ask", json={"question": "a"}).status_code == 422
    assert client.post("/ask", json={"question": "kira", "top_k": 0}).status_code == 422
