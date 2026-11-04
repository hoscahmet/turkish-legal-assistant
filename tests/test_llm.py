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
def fake(monkeypatch):
    def install(*responses):
        messages = FakeMessages(responses)
        monkeypatch.setattr(llm, "get_client", lambda: SimpleNamespace(beta=SimpleNamespace(messages=messages)))
        return messages

    return install


def test_tool_loop_returns_structured_answer(fake):
    messages = fake(
        calls_tools(tool_use("search_articles", query="yıllık izin süresi", law_number="4857")),
        final(
            "Bir ila beş yıl çalışan işçinin yıllık izni en az on dört gündür.",
            [{"law_number": "4857", "article_number": "53"}],
        ),
    )
    result = llm.answer("Yıllık izin kaç gündür?")

    assert result.answer.startswith("Bir ila beş yıl")
    assert [(s.law_number, s.article_number) for s in result.sources] == [("4857", "53")]
    # The second request carries the tool result for the first request's tool call.
    tool_result = messages.calls[1]["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "toolu_search_articles"
    assert "Yıllık ücretli izin" in tool_result["content"]
    assert tool_result["is_error"] is False


def test_hallucinated_citations_are_dropped(fake):
    fake(final("Cevap.", [{"law_number": "4857", "article_number": "53"}, {"law_number": "4857", "article_number": "999"}]))
    assert [s.article_number for s in llm.answer("soru").sources] == ["53"]


def test_unknown_article_is_reported_as_tool_error(fake):
    messages = fake(calls_tools(tool_use("get_article", law_number="6698", article_number="99")), final("Bulunamadı.", []))
    llm.answer("KVKK 99. madde ne diyor?")
    assert messages.calls[1]["messages"][-1]["content"][0]["is_error"] is True


def test_refusal_returns_turkish_message(fake):
    fake(SimpleNamespace(stop_reason="refusal", content=[]))
    result = llm.answer("soru")
    assert result.answer == llm.REFUSAL_ANSWER
    assert result.sources == []


def test_tool_loop_is_capped(fake):
    fake(*[calls_tools(tool_use("search_articles", query="kira"))] * llm.MAX_TURNS)
    with pytest.raises(llm.LLMError):
        llm.answer("soru")


def test_run_tool_search_respects_law_filter():
    content, is_error = llm.run_tool("search_articles", {"query": "kira bedeli", "law_number": "6098"}, top_k=3)
    assert not is_error
    assert {a["law_number"] for a in json.loads(content)} == {"6098"}
