import pytest
from fastapi.testclient import TestClient
from langchain_core.exceptions import ModelAPIError, ModelAuthenticationError, ModelRateLimitError, ModelRefusalError

from turkish_legal_assistant import agent
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


def test_ask_uses_agent(monkeypatch):
    fake = AskResponse(question="soru", answer="Cevap.", sources=[])
    monkeypatch.setattr(agent, "answer", lambda question, top_k: fake)
    r = client.post("/ask", json={"question": "Yıllık izin kaç gündür?"})
    assert r.status_code == 200
    assert r.json()["answer"] == "Cevap."


@pytest.mark.parametrize(
    "error, status",
    [
        (agent.LLMNotConfigured("set GEMINI_API_KEY"), 503),
        (ModelAuthenticationError("invalid key"), 503),
        (ModelRateLimitError("slow down"), 503),
        (ModelAPIError("server error"), 502),
        (agent.LLMError("no answer"), 502),
    ],
)
def test_ask_maps_llm_errors(monkeypatch, error, status):
    def fail(question, top_k):
        raise error

    monkeypatch.setattr(agent, "answer", fail)
    assert client.post("/ask", json={"question": "Yıllık izin?"}).status_code == status


def test_ask_refusal_returns_turkish_message(monkeypatch):
    def refuse(question, top_k):
        raise ModelRefusalError()

    monkeypatch.setattr(agent, "answer", refuse)
    r = client.post("/ask", json={"question": "Yıllık izin?"})
    assert r.status_code == 200
    assert r.json()["answer"] == "Bu soruya cevap veremiyorum."


def test_ask_validates_input():
    assert client.post("/ask", json={"question": "a"}).status_code == 422
    assert client.post("/ask", json={"question": "kira", "top_k": 0}).status_code == 422


def test_demo_page():
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Mevzuat Asistanı" in r.text


@pytest.mark.parametrize(
    ("provider", "model"),
    [("gemini", agent.GEMINI_MODEL), ("claude", agent.CLAUDE_MODEL), ("local", "fine-tuned Qwen3 student (vLLM)")],
)
def test_info_reports_model(monkeypatch, provider, model):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    assert client.get("/info").json() == {"llm_provider": provider, "model": model}
