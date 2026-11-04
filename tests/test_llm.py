import json
from types import SimpleNamespace

import pytest

from turkish_legal_assistant import llm


class FakeMessages:
    """Stands in for client.beta.messages: returns scripted responses and records each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def tool_use(name, **args):
    return SimpleNamespace(type="tool_use", id=f"toolu_{name}", name=name, input=args)


def final(answer, citations):
    text = json.dumps({"answer": answer, "citations": citations}, ensure_ascii=False)
    return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


def calls_tools(*blocks):
    return SimpleNamespace(stop_reason="tool_use", content=list(blocks))


@pytest.fixture
def fake_claude(monkeypatch):
    def install(*responses):
        messages = FakeMessages(responses)
        monkeypatch.setattr(llm, "get_claude_client", lambda: SimpleNamespace(beta=SimpleNamespace(messages=messages)))
        return messages

    return install


def test_tool_loop_returns_structured_answer(fake_claude):
    messages = fake_claude(
        calls_tools(tool_use("search_articles", query="yıllık izin süresi", law_number="4857")),
        final(
            "Bir ila beş yıl çalışan işçinin yıllık izni en az on dört gündür.",
            [{"law_number": "4857", "article_number": "53"}],
        ),
    )
    result = llm.answer_claude("Yıllık izin kaç gündür?")

    assert result.answer.startswith("Bir ila beş yıl")
    assert [(s.law_number, s.article_number) for s in result.sources] == [("4857", "53")]
    # The second request carries the tool result for the first request's tool call.
    tool_result = messages.calls[1]["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "toolu_search_articles"
    assert "Yıllık ücretli izin" in tool_result["content"]
    assert tool_result["is_error"] is False


def test_hallucinated_citations_are_dropped(fake_claude):
    fake_claude(final("Cevap.", [{"law_number": "4857", "article_number": "53"}, {"law_number": "4857", "article_number": "999"}]))
    assert [s.article_number for s in llm.answer_claude("soru").sources] == ["53"]


def test_unknown_article_is_reported_as_tool_error(fake_claude):
    messages = fake_claude(calls_tools(tool_use("get_article", law_number="6698", article_number="99")), final("Bulunamadı.", []))
    llm.answer_claude("KVKK 99. madde ne diyor?")
    assert messages.calls[1]["messages"][-1]["content"][0]["is_error"] is True


def test_refusal_returns_turkish_message(fake_claude):
    fake_claude(SimpleNamespace(stop_reason="refusal", content=[]))
    result = llm.answer_claude("soru")
    assert result.answer == llm.REFUSAL_ANSWER
    assert result.sources == []


def test_tool_loop_is_capped(fake_claude):
    fake_claude(*[calls_tools(tool_use("search_articles", query="kira"))] * llm.MAX_TURNS)
    with pytest.raises(llm.LLMError):
        llm.answer_claude("soru")


def test_run_tool_search_respects_law_filter():
    content, is_error = llm.run_tool("search_articles", {"query": "kira bedeli", "law_number": "6098"}, top_k=3)
    assert not is_error
    assert {a["law_number"] for a in json.loads(content)} == {"6098"}


# --- Gemini ---

from google.genai import types  # noqa: E402


class FakeGeminiModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, **kwargs):
        # Copy the list: the loop keeps appending to the same contents object.
        self.calls.append({**kwargs, "contents": list(kwargs["contents"])})
        return self.responses.pop(0)


def gemini_calls(*calls):
    content = types.Content(role="model", parts=[types.Part(function_call=c) for c in calls])
    return SimpleNamespace(function_calls=list(calls), candidates=[SimpleNamespace(content=content)], text=None)


def gemini_final(answer, citations):
    text = json.dumps({"answer": answer, "citations": citations}, ensure_ascii=False)
    return SimpleNamespace(function_calls=None, candidates=[SimpleNamespace(finish_reason=None)], text=text)


@pytest.fixture
def fake_gemini(monkeypatch):
    def install(*responses):
        models = FakeGeminiModels(responses)
        monkeypatch.setattr(llm, "get_gemini_client", lambda: SimpleNamespace(models=models))
        return models

    return install


def test_gemini_tool_loop_returns_structured_answer(fake_gemini):
    call = types.FunctionCall(id="call_1", name="search_articles", args={"query": "depozito güvence", "law_number": "6098"})
    models = fake_gemini(
        gemini_calls(call),
        gemini_final("Güvence en fazla üç aylık kira bedeli olabilir.", [{"law_number": "6098", "article_number": "342"}, {"law_number": "6098", "article_number": "1"}]),
    )
    result = llm.answer_gemini("Depozito en fazla ne kadar olabilir?")

    assert [(s.law_number, s.article_number) for s in result.sources] == [("6098", "342")]
    function_response = models.calls[1]["contents"][-1].parts[0].function_response
    assert (function_response.id, function_response.name) == ("call_1", "search_articles")
    assert "üç aylık kira" in function_response.response["result"]


def test_gemini_blocked_answer_returns_turkish_message(fake_gemini):
    blocked = SimpleNamespace(function_calls=None, candidates=[SimpleNamespace(finish_reason=types.FinishReason.SAFETY)], text=None)
    fake_gemini(blocked)
    assert llm.answer_gemini("soru").answer == llm.REFUSAL_ANSWER


def test_gemini_without_key_is_not_configured(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    llm.get_gemini_client.cache_clear()
    with pytest.raises(llm.LLMNotConfigured):
        llm.get_gemini_client()


def test_provider_is_chosen_by_env(monkeypatch, fake_gemini, fake_claude):
    fake_gemini(gemini_final("Gemini cevabı.", []))
    fake_claude(final("Claude cevabı.", []))
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm.answer("soru").answer == "Gemini cevabı."
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    assert llm.answer("soru").answer == "Claude cevabı."
