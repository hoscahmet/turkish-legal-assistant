import asyncio

from mcp import Client

from turkish_legal_assistant.mcp_server import server


def run(coro_fn):
    """Connect an in-process MCP client, so tests go through the real protocol."""

    async def main():
        async with Client(server) as client:
            return await coro_fn(client)

    return asyncio.run(main())


def call(name, args=None):
    return run(lambda c: c.call_tool(name, args or {}))


def test_lists_read_only_tools():
    tools = run(lambda c: c.list_tools()).tools
    assert {t.name for t in tools} == {"search_articles", "get_article", "list_laws"}
    assert all(t.annotations.read_only_hint for t in tools)
    search = next(t for t in tools if t.name == "search_articles")
    assert search.input_schema["required"] == ["query"]


def test_search_articles_with_law_filter():
    # conftest makes retrieval fall back to keyword search, so no model is loaded.
    result = call("search_articles", {"query": "kira bedeli", "law_number": "6098", "top_k": 3})
    articles = result.structured_content["result"]
    assert not result.is_error
    assert len(articles) == 3
    assert {a["law_number"] for a in articles} == {"6098"}


def test_get_article():
    result = call("get_article", {"law_number": "4857", "article_number": "53"})
    assert result.structured_content["title"] == "Yıllık ücretli izin hakkı ve izin süreleri"


def test_missing_article_is_a_tool_error():
    result = call("get_article", {"law_number": "4857", "article_number": "9999"})
    assert result.is_error
    assert "not found" in result.content[0].text


def test_invalid_law_number_is_rejected():
    assert call("search_articles", {"query": "kira", "law_number": "1234"}).is_error


def test_list_laws():
    laws = call("list_laws").structured_content["result"]
    assert {law["number"] for law in laws} == {"4857", "6698", "6502", "6098"}
