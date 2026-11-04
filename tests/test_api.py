import anthropic
import httpx2
from fastapi.testclient import TestClient

from turkish_legal_assistant import llm
from turkish_legal_assistant.main import app
from turkish_legal_assistant.models import AskResponse

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


def test_get_article_with_slash_and_prefix():
    assert client.get("/laws/6502/articles/47/A").json()["title"] == "Doğrudan satış sistemi"
    assert client.get("/laws/6698/articles/Geçici 1").json()["title"] == "Geçiş hükümleri"


def test_get_article_not_found():
    assert client.get("/laws/4857/articles/9999").status_code == 404


def test_ask_uses_llm(monkeypatch):
    fake = AskResponse(question="soru", answer="Cevap.", sources=[])
    monkeypatch.setattr(llm, "answer", lambda question, top_k: fake)
    r = client.post("/ask", json={"question": "Yıllık izin kaç gündür?"})
    assert r.status_code == 200
    assert r.json()["answer"] == "Cevap."


def test_ask_without_api_key_returns_503(monkeypatch):
    def fail(question, top_k):
        request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        raise anthropic.AuthenticationError("invalid x-api-key", response=httpx2.Response(401, request=request), body=None)

    monkeypatch.setattr(llm, "answer", fail)
    assert client.post("/ask", json={"question": "Yıllık izin?"}).status_code == 503


def test_ask_validates_input():
    assert client.post("/ask", json={"question": "a"}).status_code == 422
    assert client.post("/ask", json={"question": "kira", "top_k": 0}).status_code == 422
