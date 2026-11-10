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
    train = ft.load_split("train", render, eos="<eos>")
    evaluation = ft.load_split("eval", render, eos="<eos>")
    assert len(train) > 10 * len(evaluation) / 2  # roughly 90/10
    for row in train + evaluation:
        answer = json.loads(row["completion"].removesuffix("<eos>"))
        assert answer["answer"] and "Soru: " in row["prompt"]


def test_eval_articles_are_unseen_in_train():
    sources = {name: {json.loads(line)["source"] for line in (OUT_DIR / f"{name}.jsonl").read_text().splitlines()}
               for name in ("train", "eval")}
    assert not sources["train"] & sources["eval"]
