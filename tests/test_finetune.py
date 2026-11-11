import json

from turkish_legal_assistant import finetune as ft
from turkish_legal_assistant.finetune_data import OUT_DIR


def render(messages):
    return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


def test_to_prompt_completion_trains_on_the_answer_only():
    example = {
        "messages": [
            {"role": "system", "content": "Sistem."},
            {"role": "user", "content": "Soru: Kaç gün iznim var?"},
            {"role": "assistant", "content": '{"answer": "14 gün.", "citations": []}'},
        ]
    }
    row = ft.to_prompt_completion(example, render, eos="<eos>")
    assert row == {
        "prompt": "<system>Sistem.<user>Soru: Kaç gün iznim var?<assistant>",
        "completion": '{"answer": "14 gün.", "citations": []}<eos>',
    }


def test_committed_splits_load():
    """The generated data in data/finetune/ is committed; every example must convert."""
    train = ft.load_split(ft.train_file(), render, eos="<eos>")
    verified = ft.load_split(ft.train_file(data=2), render, eos="<eos>")
    evaluation = ft.load_split(OUT_DIR / "eval.jsonl", render, eos="<eos>")
    assert len(train) > 10 * len(evaluation) / 2  # roughly 90/10
    assert len(verified) > len(train)
    for row in train + verified + evaluation:
        answer = json.loads(row["completion"].removesuffix("<eos>"))
        assert answer["answer"] and "Soru: " in row["prompt"]


def test_eval_articles_are_unseen_in_train():
    files = {"train": ft.train_file(), "verified": ft.train_file(data=2), "eval": OUT_DIR / "eval.jsonl"}
    sources = {name: {json.loads(line)["source"] for line in path.read_text().splitlines()} for name, path in files.items()}
    assert not sources["train"] & sources["eval"]
    assert not sources["verified"] & sources["eval"]


def test_each_data_version_and_epoch_count_has_its_own_adapter():
    assert ft.adapter_dir("4b").name == "qwen3-4b-legal-lora"
    assert ft.adapter_dir("4b", epochs=1).name == "qwen3-4b-1ep-legal-lora"
    assert ft.adapter_dir("4b", data=2).name == "qwen3-4b-v2-legal-lora"
    assert ft.train_file(data=2).parent.name == "v2"
