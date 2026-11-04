import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from turkish_legal_assistant import agent
from turkish_legal_assistant.data import find_article
from turkish_legal_assistant.models import Citation, LegalAnswer


class FakeModel:
    """Scripted chat model: agent turns come from `turns`, the respond step returns `final`."""

    def __init__(self, turns, final):
        self.turns = list(turns)
        self.final = final
        self.agent_inputs = []

    def bind_tools(self, tools):
        def invoke(messages):
            self.agent_inputs.append(messages)
            return self.turns.pop(0)

        return SimpleNamespace(invoke=invoke)

    def with_structured_output(self, schema):
        return SimpleNamespace(invoke=lambda messages: self.final)


def calls(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])


def answer_citing(*keys):
    return LegalAnswer(answer="Cevap.", citations=[Citation(law_number=l, article_number=a) for l, a in keys])


@pytest.fixture
def use_model(monkeypatch):
    def install(model):
        monkeypatch.setattr(agent, "get_graph", lambda: agent.build_graph(model))
        return model

    return install


def tool_messages(messages):
    return [m for m in messages if isinstance(m, ToolMessage)]


def test_agent_searches_then_answers(use_model):
    model = use_model(
        FakeModel(
            [calls("search_articles", query="yıllık izin süresi", law_number="4857"), AIMessage("Buldum.")],
            answer_citing(("4857", "53"), ("4857", "999")),
        )
    )
    result = agent.answer("Yıllık izin kaç gündür?", top_k=2)

    # The hallucinated citation (art. 999) is dropped.
    assert [(s.law_number, s.article_number) for s in result.sources] == [("4857", "53")]
    # The second agent turn sees the tool result, limited to the request's top_k.
    [tool_result] = tool_messages(model.agent_inputs[1])
    articles = json.loads(tool_result.content)
    assert len(articles) == 2
    assert articles[0]["article_number"] == "53"


def test_unknown_article_is_reported_to_the_model(use_model):
    model = use_model(FakeModel([calls("get_article", law_number="6698", article_number="99"), AIMessage("Yok.")], answer_citing()))
    agent.answer("KVKK 99. madde?")
    assert tool_messages(model.agent_inputs[1])[0].content.startswith("Error")


def test_tool_loop_is_capped(use_model):
    # Fresh messages each turn: LangGraph merges messages that share an id.
    use_model(FakeModel([calls("search_articles", query="kira") for _ in range(20)], answer_citing()))
    with pytest.raises(agent.LLMError):
        agent.answer("soru")


def test_missing_structured_answer_is_an_error(use_model):
    use_model(FakeModel([AIMessage("Cevap.")], final=None))
    with pytest.raises(agent.LLMError):
        agent.answer("soru")


def test_model_needs_gemini_key(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    agent.get_model.cache_clear()
    with pytest.raises(agent.LLMNotConfigured):
        agent.get_model()


def test_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    agent.get_model.cache_clear()
    with pytest.raises(agent.LLMNotConfigured):
        agent.get_model()
    agent.get_model.cache_clear()


def test_search_tool_respects_law_filter():
    content = agent.search_articles.invoke({"query": "kira bedeli", "law_number": "6098"})
    assert {a["law_number"] for a in json.loads(content)} == {"6098"}


def test_retrieve_uses_dense_index_when_available(monkeypatch):
    article = find_article("6098", "342")
    modes = []

    class FakeIndex:
        def search(self, query, k, mode, law_number):
            modes.append(mode)
            return [(article, 0.9)]

    monkeypatch.setattr(agent.vectors, "get_index", lambda: FakeIndex())
    assert agent.retrieve("depozito", 3) == [article]
    assert modes == ["dense"]
