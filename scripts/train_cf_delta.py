#!/usr/bin/env python3
"""Phase-5 QLoRA with final-assistant-only supervision for CF-DeltaTIME."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import unsloth  # noqa: F401 - must patch before transformers/TRL imports
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)
from unsloth import FastLanguageModel

from trainer_pkg.chat_template import CHAT_TEMPLATE


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Merged TIME checkpoint used as Phase-5 base.")
    parser.add_argument("--train-data", required=True, help="JSON conversations ending in an assistant target.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--limit", type=int, help="Optional deterministic prefix for a pipeline smoke test.")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def tokenize_final_assistant_only(tokenizer, conversations: list[list[dict]], max_length: int) -> Dataset:
    rows = []
    truncated = 0
    empty_targets = 0
    for index, conversation in enumerate(conversations):
        if not conversation or conversation[-1].get("role") != "assistant":
            raise ValueError(f"conversation {index} must end with an assistant target")
        prompt_text = tokenizer.apply_chat_template(
            conversation[:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(f"chat-template prefix mismatch at conversation {index}")
        if len(full_ids) > max_length:
            truncated += 1
            left_trim = len(full_ids) - max_length
            full_ids = full_ids[left_trim:]
            prompt_len = max(0, len(prompt_ids) - left_trim)
        else:
            prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        if all(label == -100 for label in labels):
            empty_targets += 1
        rows.append(
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        )
    if empty_targets:
        raise RuntimeError(f"{empty_targets} examples lost their entire assistant target")
    lengths = [len(x["input_ids"]) for x in rows]
    target_lengths = [sum(label != -100 for label in x["labels"]) for x in rows]
    print(
        f"Tokenized {len(rows)} examples | max={max(lengths)} mean={np.mean(lengths):.1f} "
        f"target_mean={np.mean(target_lengths):.1f} truncated={truncated}"
    )
    return Dataset.from_list(rows)


def main() -> None:
    cfg = args()
    seed_everything(cfg.seed)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    conversations = json.loads(Path(cfg.train_data).read_text())
    if cfg.limit:
        conversations = conversations[: cfg.limit]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokenizer.chat_template = CHAT_TEMPLATE
    dataset = tokenize_final_assistant_only(tokenizer, conversations, cfg.max_length)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model,
        max_seq_length=cfg.max_length,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=cfg.seed,
    )

    resolved = vars(cfg) | {
        "train_examples": len(dataset),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
    }
    (output / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output / "trainer"),
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            warmup_ratio=0.05,
            logging_steps=5,
            save_strategy="epoch",
            save_total_limit=2,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            max_grad_norm=1.0,
            report_to="none",
            seed=cfg.seed,
            bf16=True,
            remove_unused_columns=False,
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
        processing_class=tokenizer,
    )
    stats = trainer.train()
    adapter = output / "final_adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    metrics = dict(stats.metrics)
    metrics["peak_vram_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    (output / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved Phase-5 adapter to {adapter}")


if __name__ == "__main__":
    main()
