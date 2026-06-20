#!/usr/bin/env python3
"""Finetune a local CENO DNA language model on CDS FASTA."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from htt_delta.model_io import import_local_ceno


class CDSFastaDataset(Dataset):
    def __init__(self, fasta_path: Path, tokenizer, max_length: int, max_samples: int | None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = self._read_fasta(fasta_path)
        if max_samples is not None and len(self.sequences) > max_samples:
            random.seed(42)
            self.sequences = random.sample(self.sequences, max_samples)

    @staticmethod
    def _read_fasta(path: Path) -> list[str]:
        sequences = []
        header = None
        parts = []
        with path.open() as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line.startswith(">"):
                    if header is not None:
                        sequence = "".join(parts).upper()
                        if len(sequence) >= 50:
                            sequences.append(sequence)
                    header = line[1:]
                    parts = []
                else:
                    parts.append(line)
        if header is not None:
            sequence = "".join(parts).upper()
            if len(sequence) >= 50:
                sequences.append(sequence)
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = self.sequences[idx][:self.max_length]
        input_ids = torch.tensor(
            self.tokenizer.encode(sequence, add_special_tokens=True),
            dtype=torch.long,
        )
        return {"input_ids": input_ids, "labels": input_ids.clone()}


class CausalLMCollator:
    def __init__(self, pad_token_id: int = 1, max_length: int = 8192):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, features):
        input_ids = [feature["input_ids"][:self.max_length] for feature in features]
        padded = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = padded.clone()
        labels[labels == self.pad_token_id] = -100
        attention_mask = (padded != self.pad_token_id).long()
        return {"input_ids": padded, "labels": labels, "attention_mask": attention_mask}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-code-dir", required=True, type=Path)
    parser.add_argument("--train-fasta", required=True, type=Path)
    parser.add_argument("--val-fasta", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--max-val-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    CENOConfig, CENOForCausalLM, CENOCharLevelTokenizer = import_local_ceno(
        args.model_code_dir
    )
    tokenizer = CENOCharLevelTokenizer(vocab_size=512)
    config = CENOConfig.from_pretrained(args.model_path)
    config.use_mamba_kernels = False
    model = CENOForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    train_dataset = CDSFastaDataset(args.train_fasta, tokenizer, args.max_length, args.max_train_samples)
    val_dataset = CDSFastaDataset(args.val_fasta, tokenizer, args.max_length, args.max_val_samples)
    n_gpu = int(os.environ.get("WORLD_SIZE", "1"))

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_dir=str(args.output_dir / "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CausalLMCollator(max_length=args.max_length),
    )
    train_result = trainer.train()
    best_model_path = args.output_dir / "best_model"
    trainer.save_model(best_model_path)
    config.save_pretrained(best_model_path)
    eval_result = trainer.evaluate()

    metrics = {
        "train_loss": train_result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "n_gpu": n_gpu,
        "effective_batch_size": args.batch_size * args.grad_accum * n_gpu,
        "best_model_path": str(best_model_path),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "training_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

