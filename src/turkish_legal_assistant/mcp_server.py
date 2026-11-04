"""MCP server: exposes article search to any MCP client (Claude Desktop, Claude Code, IDEs).

The client's own model does the reasoning, so no LLM API key is needed here.
    uv run python -m turkish_legal_assistant.mcp_server   (stdio transport)
"""

from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import Field

from .agent import LawNumber, retrieve
from .data import LAWS, find_article
from .models import Article, Law

INSTRUCTIONS = """Turkish legislation: 4857 İş Kanunu, 6698 KVKK, 6502 Tüketicinin Korunması Hakkında Kanun,
6098 Türk Borçlar Kanunu (lease section, articles 299-378). Search with Turkish legal terms,
answer in Turkish and cite law and article numbers. This is general information, not legal advice."""

server = MCPServer("turkish-legal-assistant", instructions=INSTRUCTIONS)
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


@server.tool(annotations=READ_ONLY)
def search_articles(
    query: Annotated[str, Field(description="What to look for, in Turkish.")],
    law_number: Annotated[LawNumber | None, Field(description="Restrict the search to one law.")] = None,
    top_k: Annotated[int, Field(ge=1, le=10)] = 5,
) -> list[Article]:
    """Semantic search over the articles. Returns the most relevant ones with full text.
    If results look off-topic, rephrase with legal terms (e.g. 'işveren' for 'patron')."""
    return retrieve(query, top_k, law_number)


@server.tool(annotations=READ_ONLY)
def get_article(law_number: LawNumber, article_number: str) -> Article:
    """Full text of one article, e.g. law_number '4857' with article_number '53', '47/A' or 'Geçici 1'."""
    article = find_article(law_number, article_number)
    if article is None:
        raise ToolError(f"Article {article_number} of law {law_number} not found.")
    return article


@server.tool(annotations=READ_ONLY)
def list_laws() -> list[Law]:
    """The laws this server covers."""
    return LAWS


if __name__ == "__main__":
    server.run()
