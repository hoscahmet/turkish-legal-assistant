"""Serve the fine-tuned student with vLLM's OpenAI-compatible server, for LLM_PROVIDER=local.

  uv run python -m turkish_legal_assistant.serve [--size 1.7b|4b] [--port 8001]

The base model is loaded once and the LoRA adapter is applied on top under the name "legal" (no merging).
4B in bf16 (8 GB) leaves too little KV cache on a 12 GB GPU, so it is quantized to 4 bits on load:
the same NF4 format it was trained against with QLoRA.
Run the API with EMBED_DEVICE=cpu so BGE-M3 does not compete with vLLM for GPU memory.
"""

import argparse
import os
import sys
from pathlib import Path

from .finetune import STUDENTS, adapter_dir
from .finetune_data import MAX_MODEL_LEN
from .local import SERVED_NAME

DEFAULT_PORT = 8001  # 8000 is uvicorn's default


def serve_command(size: str, port: int = DEFAULT_PORT) -> list[str]:
    adapter = adapter_dir(size)
    if not adapter.exists():
        raise SystemExit(f"{adapter} missing: train it with `python -m turkish_legal_assistant.finetune --size {size}`")
    command = [
        str(Path(sys.executable).parent / "vllm"),  # the vllm CLI installed in this virtualenv
        "serve",
        STUDENTS[size],
        "--port", str(port),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.75",
        "--enable-lora",
        "--max-lora-rank", "16",
        "--lora-modules", f"{SERVED_NAME}={adapter}",
    ]
    if size == "4b":
        command += ["--quantization", "bitsandbytes"]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=STUDENTS, default="4b")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    command = serve_command(args.size, args.port)
    print(" ".join(command))
    # See finetune_data.vllm_generate: no nvcc in this WSL, so no FlashInfer sampler.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.execv(command[0], command)  # become the vLLM process, so Ctrl+C stops it directly


if __name__ == "__main__":
    main()
