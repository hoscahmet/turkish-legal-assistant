# API + demo UI. The fine-tuning stack (vLLM, peft, trl) is left out; the student runs in the vllm service.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1

WORKDIR /app

# Dependencies first, in their own layer, so code changes do not reinstall them.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-group finetune --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-group finetune

# data/ (articles + embedding cache) is mounted at runtime, see compose.yaml.
ENV PATH="/app/.venv/bin:$PATH" EMBED_DEVICE=cpu
EXPOSE 8000
CMD ["uvicorn", "turkish_legal_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]
