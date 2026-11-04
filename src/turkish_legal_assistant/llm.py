"""Answer questions with an LLM: tool calling to find articles, structured output for the answer.

Two providers run the same loop: Gemini (free tier, default) and Claude. Pick with LLM_PROVIDER.
"""

import json
import os
from functools import cache

import anthropic
from google import genai
from google.genai import types

from .data import find_article
from .ingest import LAWS
from .models import AskResponse, LegalAnswer
from .search import search

CLAUDE_MODEL = "claude-opus-5-5"
GEMINI_MODEL = "gemini-3.8-flash"  # has a free tier
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


class LLMNotConfigured(LLMError):
    pass


# Clients are created on first use so the app (and tests) start without an API key.
@cache
def get_claude_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


@cache
def get_gemini_client() -> genai.Client:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise LLMNotConfigured("set GEMINI_API_KEY")
    return genai.Client()


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
    provider = os.environ.get("LLM_PROVIDER", "gemini")
    if provider == "gemini":
        return answer_gemini(question, top_k)
    if provider == "claude":
        return answer_claude(question, top_k)
    raise LLMNotConfigured(f"Unknown LLM_PROVIDER: {provider}")


def to_response(question: str, json_text: str) -> AskResponse:
    """Validate the model's JSON answer and keep only citations of real articles (hallucination guard)."""
    parsed = LegalAnswer.model_validate_json(json_text)
    sources = [a for c in parsed.citations if (a := find_article(c.law_number, c.article_number))]
    return AskResponse(question=question, answer=parsed.answer, sources=sources)


def answer_claude(question: str, top_k: int = 5) -> AskResponse:
    messages: list = [{"role": "user", "content": question}]
    for _ in range(MAX_TURNS):
        response = get_claude_client().beta.messages.create(
            model=CLAUDE_MODEL,
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

    return to_response(question, next(b.text for b in response.content if b.type == "text"))


GEMINI_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM,
    tools=[
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"], description=t["description"], parameters_json_schema=t["input_schema"]
                )
                for t in TOOLS
            ]
        )
    ],
    # We run the tool loop ourselves, same as for Claude, instead of letting the SDK call Python functions.
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    response_mime_type="application/json",
    response_json_schema=LegalAnswer.model_json_schema(),
)
BLOCKED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}


def answer_gemini(question: str, top_k: int = 5) -> AskResponse:
    contents: list[types.Content] = [types.Content(role="user", parts=[types.Part(text=question)])]
    for _ in range(MAX_TURNS):
        response = get_gemini_client().models.generate_content(
            model=GEMINI_MODEL, contents=contents, config=GEMINI_CONFIG
        )
        if not response.function_calls:
            break
        # Send the model turn back unchanged (it carries thought signatures), then all tool results.
        contents.append(response.candidates[0].content)
        parts = []
        for call in response.function_calls:
            content, is_error = run_tool(call.name, call.args or {}, top_k)
            result = types.FunctionResponse(
                id=call.id, name=call.name, response={"error" if is_error else "result": content}
            )
            parts.append(types.Part(function_response=result))
        contents.append(types.Content(role="user", parts=parts))
    else:
        raise LLMError(f"No answer after {MAX_TURNS} tool-calling rounds")

    if not response.text:
        reason = response.candidates[0].finish_reason if response.candidates else None
        if not response.candidates or getattr(reason, "name", None) in BLOCKED:
            return AskResponse(question=question, answer=REFUSAL_ANSWER, sources=[])
        raise LLMError(f"Empty answer, finish reason: {reason}")
    return to_response(question, response.text)
