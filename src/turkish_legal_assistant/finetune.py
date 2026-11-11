"""QLoRA fine-tune of a Qwen3 student (1.7B or 4B) on the teacher's answers in data/finetune/.

  uv run python -m turkish_legal_assistant.finetune [--size 1.7b|4b] [--epochs N] [--data 1|2]

--data 2 trains on the verified set in data/finetune/v2/ (see verify.py) and saves a separate "-v2" adapter.

The base model is loaded in 4 bits and frozen; only small LoRA matrices are trained, so a 1.7B or 4B model fits
in 12 GB with room for 5k-token prompts. Loss is computed on the answer only: the student should learn to
write the LegalAnswer JSON, not to reproduce the articles in its prompt.
The adapter is saved to models/ (gitignored); vLLM can serve it on top of the base model without merging.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from .finetune_data import OUT_DIR, Messages

# Student sizes that fit a 12 GB GPU for QLoRA training and bf16 serving. 8B is the teacher itself.
STUDENTS = {"1.7b": "Qwen/Qwen3-1.7B", "4b": "Qwen/Qwen3-4B"}
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


DEFAULT_EPOCHS = 3


DATA_VERSIONS = (1, 2)  # 1: every teacher answer (finetune_data.py); 2: judge-verified and larger (verify.py)


def adapter_dir(size: str, epochs: int = DEFAULT_EPOCHS, data: int = 1) -> Path:
    """The default run keeps the plain name; other epoch counts and data versions get their own folder."""
    suffix = ("" if data == 1 else f"-v{data}") + ("" if epochs == DEFAULT_EPOCHS else f"-{epochs}ep")
    return MODELS_DIR / f"qwen3-{size}{suffix}-legal-lora"


def train_file(data: int = 1) -> Path:
    return OUT_DIR / "train.jsonl" if data == 1 else OUT_DIR / f"v{data}" / "train.jsonl"


# The longest example is 5,327 tokens. Truncating would cut the answer at the end, so nothing is truncated.
MAX_LENGTH = 5376


def to_prompt_completion(example: dict, render: Callable[[Messages], str], eos: str) -> dict:
    """Split a chat example into the rendered prompt and the answer the loss is computed on.

    `render` applies the chat template with thinking off, so training sees exactly the prompt that
    inference sends (it ends with an empty <think></think> block).
    """
    *prompt, answer = example["messages"]
    assert answer["role"] == "assistant"
    return {"prompt": render(prompt), "completion": answer["content"] + eos}


def load_split(path: Path, render: Callable[[Messages], str], eos: str) -> list[dict]:
    lines = path.read_text().splitlines()
    return [to_prompt_completion(json.loads(line), render, eos) for line in lines if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=STUDENTS, default="1.7b")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--data", type=int, choices=DATA_VERSIONS, default=1)
    args = parser.parse_args()
    epochs, data = args.epochs, args.data
    student, out_dir = STUDENTS[args.size], adapter_dir(args.size, epochs, data)

    # Heavy imports here, so tests can import this module without a GPU stack.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(student)

    def render(messages: Messages) -> str:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    train = Dataset.from_list(load_split(train_file(data), render, tokenizer.eos_token))
    evaluation = Dataset.from_list(load_split(OUT_DIR / "eval.jsonl", render, tokenizer.eos_token))

    model = AutoModelForCausalLM.from_pretrained(
        student,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # 4-bit format suited to normally distributed weights (the "Q" in QLoRA)
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",  # attention and MLP projections
        task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=str(out_dir),
        max_length=MAX_LENGTH,
        completion_only_loss=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # effective batch of 8 examples
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        gradient_checkpointing=True,  # recompute activations in the backward pass: slower, much less memory
        per_device_eval_batch_size=1,
        eval_strategy="epoch",
        save_strategy="no",  # only the final adapter is kept
        logging_steps=5,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=train, eval_dataset=evaluation, processing_class=tokenizer, peft_config=lora
    )
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"Adapter saved to {out_dir}")


if __name__ == "__main__":
    main()
