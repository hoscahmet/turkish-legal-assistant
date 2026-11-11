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
    adapter = tmp_path / "qwen3-4b-legal-lora"
    adapter.mkdir()
    monkeypatch.setattr(serve, "adapter_dir", lambda size: adapter)
    local = serve.serve_command("4b")[2:]  # drop the vllm binary and "serve"
    compose = COMPOSE["services"]["vllm"]["command"]

    assert compose[0] == local[0]  # same base model
    expected = [f.replace(str(adapter), f"/models/{adapter.name}") for f in flag_pairs(local[1:])]
    assert flag_pairs(compose[1:]) == expected
    assert f"--lora-modules={SERVED_NAME}=/models/qwen3-4b-legal-lora" in compose


def test_api_reaches_vllm_by_service_name():
    env = COMPOSE["services"]["api"]["environment"]
    assert env["LOCAL_LLM_URL"] == "http://vllm:8001/v1"
    assert env["EMBED_DEVICE"] == "cpu"  # the GPU belongs to vLLM


def test_vllm_is_opt_in():
    assert COMPOSE["services"]["vllm"]["profiles"] == ["local"]
    assert COMPOSE["services"]["api"]["depends_on"]["vllm"]["required"] is False
