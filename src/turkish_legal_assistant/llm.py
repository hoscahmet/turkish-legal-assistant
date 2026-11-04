"""Answer questions with Claude: tool calling to find articles, structured output for the answer."""

import json
from functools import cache

import anthropic

from .data import find_article
from .ingest import LAWS
from .models import AskResponse, LegalAnswer
from .search import search

MODEL = "claude-opus-5-5"
MAX_TURNS = 6  # cap tool-calling rounds so a confused model cannot loop (and bill) forever
REFUSAL_ANSWER = "Bu soruya cevap veremiyorum."

SYSTEM = """You are a legal information assistant for Turkish legislation.
- Answer in Turkish, clearly and briefly, for a non-lawyer.
- Before answering, use the tools to find the relevant articles. Base the answer only on article texts the tools returned; never on memory.
- Cite every article the answer relies on.
- If the articles you found do not answer the question, say so instead of guessing.
- You give general information, not legal advice. For case-specific questions, suggest consulting a lawyer."""

_law_list = ", ".join(f"{n} {name}" for n, (name, _) in LAWS.items())
TOOLS = [
    {
        "name": "search_articles",
        "description": (
            f"Keyword search over articles of these laws: {_law_list} (only the lease section, "
            "articles 299-378, of 6098). Returns matching articles with their full text. "
            "Matching is by exact Turkish words, so try synonyms or word forms if results are poor."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Turkish keywords."},
                "law_number": {"type": "string", "enum": list(LAWS), "description": "Restrict to one law."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_article",
        "description": "Get the full text of one article, e.g. law_number '4857', article_number '53', '47/A' or 'Geçici 1'.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "law_number": {"type": "string", "enum": list(LAWS)},
                "article_number": {"type": "string"},
            },
            "required": ["law_number", "article_number"],
            "additionalProperties": False,
        },
    },
]
ANSWER_SCHEMA = anthropic.transform_schema(LegalAnswer)


class LLMError(Exception):
    pass


@cache
def get_client() -> anthropic.Anthropic:
    # Created on first use so the app (and tests) start without an API key.
    return anthropic.Anthropic()


def run_tool(name: str, args: dict, top_k: int) -> tuple[str, bool]:
    """Execute a tool call; returns (result text, is_error)."""
    if name == "search_articles":
        hits = search(args["query"], top_k, args.get("law_number"))
        if not hits:
            return "No matching articles. Try different keywords.", False
        return json.dumps([h.article.model_dump(exclude={"source_url"}) for h in hits], ensure_ascii=False), False
    if name == "get_article":
        article = find_article(args["law_number"], args["article_number"])
        if article is None:
            return f"Article {args['article_number']} of law {args['law_number']} not found.", True
        return article.model_dump_json(exclude={"source_url"}), False
    return f"Unknown tool: {name}", True


def answer(question: str, top_k: int = 5) -> AskResponse:
    messages: list = [{"role": "user", "content": question}]
    for _ in range(MAX_TURNS):
        response = get_client().beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
            # If a safety classifier declines, the API retries on a suitable fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason != "tool_use":
            break
        # Send the whole assistant turn back unchanged, then one user turn with all tool results.
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                content, is_error = run_tool(block.name, block.input, top_k)
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error}
                )
        messages.append({"role": "user", "content": results})
    else:
        raise LLMError(f"No answer after {MAX_TURNS} tool-calling rounds")

    if response.stop_reason == "refusal":
        return AskResponse(question=question, answer=REFUSAL_ANSWER, sources=[])
    if response.stop_reason != "end_turn":
        raise LLMError(f"Unexpected stop reason: {response.stop_reason}")

    text = next(b.text for b in response.content if b.type == "text")
    parsed = LegalAnswer.model_validate_json(text)
    # Hallucination guard: only keep citations of articles that really exist.
    sources = [a for c in parsed.citations if (a := find_article(c.law_number, c.article_number))]
    return AskResponse(question=question, answer=parsed.answer, sources=sources)
