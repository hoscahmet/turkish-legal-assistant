"""LangGraph agent: the model searches articles with tools, then a final step writes a structured answer.

    START -> agent --(tool calls?)--> tools -> agent ... --(no tool calls)--> respond -> END

LLM_PROVIDER picks the model: "gemini" (default, free tier) or "claude".
LLM_PROVIDER=local skips the agent: the fine-tuned student only answers from retrieved articles (see local.py).
"""

import json
import logging
import os
from functools import cache
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from . import local, vectors
from .data import find_article
from .models import Article, AskResponse, LegalAnswer
from .search import search

log = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-opus-5-5"
# Free-tier models get overloaded at peak times; override with GEMINI_MODEL to switch.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_TOOL_ROUNDS = 6  # cap agent <-> tools cycles so a confused model cannot loop forever
LawNumber = Literal["4857", "6698", "6502", "6098"]

SYSTEM = """You are a legal information assistant for Turkish legislation.
- Answer in Turkish, clearly and briefly, for a non-lawyer.
- Before answering, use the tools to find the relevant articles. Base the answer only on article texts the tools returned; never on memory.
- Cite every article the answer relies on.
- If the articles you found do not answer the question, say so instead of guessing.
- You give general information, not legal advice. For case-specific questions, suggest consulting a lawyer."""
RESPOND = "Now write the final answer to the user's question as JSON, citing the articles you used."


class State(MessagesState):
    """Graph state: the message history plus the final structured answer."""

    answer: LegalAnswer | None


class LLMError(Exception):
    pass


class LLMNotConfigured(LLMError):
    pass


def retrieve(query: str, top_k: int, law_number: str | None = None) -> list[Article]:
    """Dense BGE-M3 search, the best retriever on the golden set (MRR 0.77 vs 0.30 for keywords)."""
    try:
        index = vectors.get_index()
    except FileNotFoundError as e:
        log.warning("%s; falling back to keyword search", e)
        return [r.article for r in search(query, top_k, law_number)]
    return [a for a, _ in index.search(query, top_k, "dense", law_number)]


def _dump(article: Article) -> dict:
    return article.model_dump(exclude={"source_url"})


@tool
def search_articles(query: str, config: RunnableConfig, law_number: LawNumber | None = None) -> str:
    """Semantic search over articles of 4857 İş Kanunu, 6698 Kişisel Verilerin Korunması Kanunu,
    6502 Tüketicinin Korunması Hakkında Kanun and 6098 Türk Borçlar Kanunu (lease section, articles 299-378 only).
    Returns the most relevant articles with their full text. If results look off-topic, rephrase with
    legal terms (e.g. 'işveren' for 'patron').

    Args:
        query: What to look for, in Turkish.
        law_number: Restrict the search to one law.
    """
    # config is injected by LangGraph (the model never sees it); it carries the request's top_k.
    top_k = config.get("configurable", {}).get("top_k", 5)
    hits = retrieve(query, top_k, law_number)
    if not hits:
        return "No matching articles. Try different keywords."
    return json.dumps([_dump(a) for a in hits], ensure_ascii=False)


@tool
def get_article(law_number: LawNumber, article_number: str) -> str:
    """Get the full text of one article, e.g. law_number '4857' with article_number '53', '47/A' or 'Geçici 1'."""
    article = find_article(law_number, article_number)
    if article is None:
        return f"Error: article {article_number} of law {law_number} not found."
    return json.dumps(_dump(article), ensure_ascii=False)


TOOLS = [search_articles, get_article]


@cache
def get_model() -> BaseChatModel:
    # Created on first use so the app (and tests) start without an API key.
    provider = os.environ.get("LLM_PROVIDER", "gemini")
    if provider == "gemini":
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise LLMNotConfigured("set GEMINI_API_KEY")
        return init_chat_model(GEMINI_MODEL, model_provider="google_genai")
    if provider == "claude":
        return init_chat_model(CLAUDE_MODEL, model_provider="anthropic", max_tokens=16000)
    raise LLMNotConfigured(f"Unknown LLM_PROVIDER: {provider}")


def build_graph(model: BaseChatModel):
    model_with_tools = model.bind_tools(TOOLS)
    structured = model.with_structured_output(LegalAnswer)

    def agent(state: State) -> dict:
        return {"messages": [model_with_tools.invoke([SystemMessage(SYSTEM), *state["messages"]])]}

    def respond(state: State) -> dict:
        # A separate call without tools: some models cannot combine tool calling and JSON output.
        answer = structured.invoke([SystemMessage(SYSTEM), *state["messages"], HumanMessage(RESPOND)])
        if answer is None:
            raise LLMError("Model returned no structured answer")
        return {"answer": answer}

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("respond", respond)
    graph.add_edge(START, "agent")
    # tools_condition returns "tools" when the last message has tool calls, END otherwise.
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "respond"})
    graph.add_edge("tools", "agent")
    graph.add_edge("respond", END)
    return graph.compile()


@cache
def get_graph():
    return build_graph(get_model())


def answer(question: str, top_k: int = 5) -> AskResponse:
    if os.environ.get("LLM_PROVIDER") == "local":
        return local.answer(question, retrieve(question, top_k))
    config = {
        "configurable": {"top_k": top_k},
        # Each round is two steps (agent, tools); +2 for the final agent and respond steps.
        "recursion_limit": 2 * MAX_TOOL_ROUNDS + 2,
    }
    try:
        state = get_graph().invoke({"messages": [HumanMessage(question)]}, config)
    except GraphRecursionError as e:
        raise LLMError(f"No answer after {MAX_TOOL_ROUNDS} tool-calling rounds") from e
    parsed = state["answer"]
    # Hallucination guard: only keep citations of articles that really exist.
    sources = [a for c in parsed.citations if (a := find_article(c.law_number, c.article_number))]
    return AskResponse(question=question, answer=parsed.answer, sources=sources)
