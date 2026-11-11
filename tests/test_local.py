import json

import httpx
import pytest
from fastapi.testclient import TestClient

from turkish_legal_assistant import agent, local, serve
from turkish_legal_assistant.data import find_article
from turkish_legal_assistant.main import app

IZIN = find_article("4857", "53")
SURE = find_article("4857", "54")


def reply(*keys, answer="İş Kanunu madde 53'e göre en az 14 gün izin hakkınız var."):
    citations = [{"law_number": law, "article_number": art} for law, art in keys]
    content = json.dumps({"answer": answer, "citations": citations}, ensure_ascii=False)
    return {"choices": [{"message": {"content": content}}]}


class FakeServer:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def __call__(self, url, payload):
        self.requests.append((url, payload))
        return self.response


def test_answer_sends_the_training_prompt_with_thinking_on_and_a_schema():
    server = FakeServer(reply(("4857", "53")))
    result = local.answer("Kaç gün iznim var?", [IZIN, SURE], post=server)

    url, payload = server.requests[0]
    assert url.endswith("/chat/completions") and payload["model"] == "legal"
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["max_tokens"] == 3072  # room for the thinking before the answer
    assert payload["response_format"]["json_schema"]["name"] == "LegalAnswer"
    user = payload["messages"][-1]["content"]
    assert "[4857/53]" in user and "[4857/54]" in user and user.endswith("Soru: Kaç gün iznim var?")
    assert result.answer.startswith("İş Kanunu madde 53") and result.sources == [IZIN]


def test_thinking_can_be_turned_off_for_the_first_adapters(monkeypatch):
    monkeypatch.setattr(local, "THINKING", False)
    payload = local.chat_payload([])
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert (payload["temperature"], payload["max_tokens"]) == (0.7, 1024)


def test_answer_drops_citations_it_was_not_given_and_duplicates():
    server = FakeServer(reply(("4857", "54"), ("4857", "999"), ("4857", "54")))
    assert local.answer("Soru?", [IZIN, SURE], post=server).sources == [SURE]


def test_answer_asks_again_once_when_the_reply_drifts_into_chinese():
    replies = [reply(("4857", "53"), answer="İşveren sorumluluk承担 eder."), reply(("4857", "53"))]
    requests = []

    def server(url, payload):
        requests.append(payload)
        return replies[len(requests) - 1]

    result = local.answer("Kaç gün iznim var?", [IZIN], post=server)
    assert len(requests) == 2 and requests[0] == requests[1]
    assert result.answer.startswith("İş Kanunu madde 53")


def test_answer_rejects_invalid_json():
    server = FakeServer({"choices": [{"message": {"content": "İzniniz 14 gün."}}]})
    with pytest.raises(local.LocalModelError):
        local.answer("Soru?", [IZIN], post=server)


def test_http_post_reports_a_missing_server(monkeypatch):
    def refuse(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", refuse)
    with pytest.raises(local.LocalModelUnavailable):
        local.http_post("http://localhost:1/v1/chat/completions", {})


def test_agent_uses_the_student_without_tools_when_provider_is_local(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setattr(agent, "get_graph", lambda: pytest.fail("the local provider must not run the agent"))
    seen = {}

    def fake_answer(question, articles):
        seen["articles"] = articles
        return local.AskResponse(question=question, answer="Cevap.", sources=[])

    monkeypatch.setattr(local, "answer", fake_answer)
    agent.answer("Yıllık izin kaç gündür?", top_k=2)
    assert len(seen["articles"]) == 2  # retrieved with the request's top_k (keyword fallback in tests)


@pytest.mark.parametrize(
    ("error", "status"), [(local.LocalModelUnavailable("down"), 503), (local.LocalModelError("bad"), 502)]
)
def test_api_maps_local_model_errors(monkeypatch, error, status):
    def fail(question, top_k):
        raise error

    monkeypatch.setattr(agent, "answer", fail)
    assert TestClient(app).post("/ask", json={"question": "Kaç gün iznim var?"}).status_code == status


def test_serve_command(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(serve, "adapter_dir", lambda size, data: seen.append((size, data)) or tmp_path)
    command = serve.serve_command("4b", port=9000)
    assert command[1:3] == ["serve", "Qwen/Qwen3-4B"]
    assert f"legal={tmp_path}" in command and "9000" in command
    assert seen == [("4b", 2)]  # the adapter trained on the verified set
    assert command[command.index("--reasoning-parser") + 1] == "qwen3"
    assert command[-2:] == ["--quantization", "bitsandbytes"]
    assert "--quantization" not in serve.serve_command("1.7b")


def test_serve_command_needs_a_trained_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "adapter_dir", lambda size, data: tmp_path / "missing")
    with pytest.raises(SystemExit):
        serve.serve_command("4b")
