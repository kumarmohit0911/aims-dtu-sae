from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["output_dir"])
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model_name"])

    dataset = load_dataset(
        data_cfg["dataset_name"],
        data_cfg.get("dataset_config"),
        split=data_cfg["split"],
        streaming=True,
    )
    dataset = dataset.select_columns([data_cfg["text_column"]]).take(data_cfg["max_examples"])

    def tokenize(batch):
        return tokenizer(
            batch[data_cfg["text_column"]],
            truncation=True,
            max_length=data_cfg["max_length"],
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=None)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_steps=train_cfg["max_steps"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", False),
        report_to="none",
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved fine-tuned model to {out_dir}")


if __name__ == "__main__":
    main()
