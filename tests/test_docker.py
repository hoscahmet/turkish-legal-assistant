import re
import tomllib
from pathlib import Path

import yaml

from turkish_legal_assistant import serve
from turkish_legal_assistant.local import SERVED_NAME

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = yaml.safe_load((ROOT / "compose.yaml").read_text())


def flag_pairs(args: list[str]) -> list[str]:
    """Normalize "--flag value" and "--flag=value" to the same "--flag=value" form."""
    out: list[str] = []
    for arg in args:
        if out and not arg.startswith("--") and "=" not in out[-1]:
            out[-1] += f"={arg}"
        else:
            out.append(arg)
    return out


def test_vllm_service_matches_serve_py(monkeypatch, tmp_path):
    # serve_command checks that the adapter exists; CI has no trained adapter, so point it at a temp dir.
    adapter = tmp_path / "qwen3-4b-v2-legal-lora"
    adapter.mkdir()
    monkeypatch.setattr(serve, "adapter_dir", lambda size, data: adapter)
    local = serve.serve_command("4b")[2:]  # drop the vllm binary and "serve"
    compose = COMPOSE["services"]["vllm"]["command"]

    assert compose[0] == local[0]  # same base model
    expected = [f.replace(str(adapter), f"/models/{adapter.name}") for f in flag_pairs(local[1:])]
    assert flag_pairs(compose[1:]) == expected
    assert f"--lora-modules={SERVED_NAME}=/models/qwen3-4b-v2-legal-lora" in compose
    assert "--reasoning-parser=qwen3" in compose  # local.py asks for thinking; the schema must apply after it


def test_api_reaches_vllm_by_service_name():
    env = COMPOSE["services"]["api"]["environment"]
    assert env["LOCAL_LLM_URL"] == "http://vllm:8001/v1"
    assert env["EMBED_DEVICE"] == "cpu"  # the GPU belongs to vLLM


def test_vllm_is_opt_in():
    assert COMPOSE["services"]["vllm"]["profiles"] == ["local"]
    assert COMPOSE["services"]["api"]["depends_on"]["vllm"]["required"] is False


def test_api_image_drops_every_cuda_wheel():
    # The Dockerfile filters torch's Linux-only (CUDA) wheels out of the locked requirements by name.
    # If a torch upgrade adds a wheel the pattern misses, the image silently grows by gigabytes.
    dockerfile = (ROOT / "Dockerfile").read_text()
    pattern = re.compile(re.search(r"grep -vE '([^']+)'", dockerfile).group(1))
    packages = {p["name"]: p for p in tomllib.loads((ROOT / "uv.lock").read_text())["package"]}

    todo = [d["name"] for d in packages["torch"]["dependencies"] if "sys_platform == 'linux'" in d.get("marker", "")]
    assert todo  # the lock still has CUDA wheels to drop
    cuda_only: set[str] = set()
    while todo:
        name = todo.pop()
        if name not in cuda_only:
            cuda_only.add(name)
            todo += [d["name"] for d in packages[name].get("dependencies", [])]

    assert [name for name in sorted(cuda_only) if not pattern.match(name)] == []
    assert "download.pytorch.org/whl/cpu" in dockerfile
