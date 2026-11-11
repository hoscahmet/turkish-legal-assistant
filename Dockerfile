# API + demo UI. The fine-tuning stack (vLLM, peft, trl) is left out; the student runs in the vllm service.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

# Dependencies first, in their own layer, so code changes do not reinstall them.
# The API embeds on the CPU (the GPU belongs to vLLM), so the CUDA wheels that torch pulls in on Linux (~4.5 GB) are
# dropped: every other package is installed at its locked version, and torch at the same version from the CPU index.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --frozen --no-dev --no-group finetune --no-emit-project --no-hashes -o /tmp/locked.txt \
    && grep -vE '^(torch|triton|nvidia-|cuda-)' /tmp/locked.txt > /tmp/cpu.txt \
    && uv venv \
    && uv pip install --no-deps -r /tmp/cpu.txt \
    && uv pip install --no-deps --index-url https://download.pytorch.org/whl/cpu "$(grep -oE '^torch==[^ ]+' /tmp/locked.txt)" \
    && rm /tmp/locked.txt /tmp/cpu.txt

COPY README.md ./
COPY src ./src
# Editable, as uv sync does it: the code finds data/ relative to its own location under /app.
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --no-deps -e .

# data/ (articles + embedding cache) is mounted at runtime, see compose.yaml.
ENV PATH="/app/.venv/bin:$PATH" EMBED_DEVICE=cpu
EXPOSE 8000
CMD ["uvicorn", "turkish_legal_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]
