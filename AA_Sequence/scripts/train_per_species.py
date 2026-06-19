#!/usr/bin/env python3
"""
ESM2 Per-Species MLM Fine-tuning — Single Species, Final Model Only

Usage:
    python train_per_species.py --species human --model_path <base_model> --data_dir data/training --output_base <output_dir>
    python train_per_species.py --species rice --model_path <base_model> --data_dir data/training --output_base <output_dir>
    python train_per_species.py --species chlamydomonas --model_path <base_model> --data_dir data/training --output_base <output_dir>

Trains one species at a time, saves only the final model (no intermediate checkpoints).
Supports DDP via HuggingFace Accelerate.
"""

import argparse, json, logging, os, random, time, math
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    EsmForMaskedLM, EsmTokenizer, AutoModelForMaskedLM, AutoTokenizer,
    Trainer, TrainingArguments, set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


# ── Dataset ────────────────────────────────────────────────────────
class ProteinDataset(Dataset):
    VALID_AAS = set("LAGVSERTIDPKQNFYMHWCXBZUO.-")
    def __init__(self, jsonl_path, min_len=10, max_len=1024, shuffle_seed=42):
        self.sequences = []
        skipped = dict(short=0, long=0, invalid=0)
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                seq = rec.get("seq", "")
                L = len(seq)
                if L < min_len:    skipped["short"] += 1; continue
                if L > max_len:    skipped["long"] += 1; continue
                if not set(seq).issubset(self.VALID_AAS): skipped["invalid"] += 1; continue
                self.sequences.append(seq)
        logger.info(f"Loaded {len(self.sequences):,} seqs "
                    f"(skipped: {skipped['short']} short, {skipped['long']} long, "
                    f"{skipped['invalid']} invalid)")
        rng = random.Random(shuffle_seed)
        rng.shuffle(self.sequences)
        lengths = [len(s) for s in self.sequences]
        logger.info(f"  Length: {min(lengths)}-{max(lengths)}, mean={np.mean(lengths):.0f}")
    def __len__(self): return len(self.sequences)
    def __getitem__(self, i): return self.sequences[i]


# ── Collator ───────────────────────────────────────────────────────
class MLMCollator:
    def __init__(self, tokenizer, mlm_prob=0.15, max_len=1024):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob
        self.max_len = max_len
        self.cls_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id
        self.mask_id = tokenizer.mask_token_id
        self.aa_to_id = {}
        for tok, tid in tokenizer.get_vocab().items():
            if len(tok) == 1 and tok.isalpha():
                self.aa_to_id[tok] = tid
        logger.info(f"AA mapping: {len(self.aa_to_id)} tokens")

    def _tokenize(self, seq):
        ids = [self.cls_id]
        for aa in seq[:self.max_len - 2]:
            ids.append(self.aa_to_id.get(aa, self.tokenizer.unk_token_id))
        ids.append(self.eos_id)
        return torch.tensor(ids, dtype=torch.long)

    def __call__(self, sequences):
        batch_ids = [self._tokenize(s) for s in sequences]
        max_len = max(len(ids) for ids in batch_ids)
        padded = torch.full((len(batch_ids), max_len), self.pad_id, dtype=torch.long)
        for i, ids in enumerate(batch_ids):
            padded[i, :len(ids)] = ids
        attention_mask = (padded != self.pad_id).long()

        # MLM masking (80/10/10 split, only on non-special tokens)
        labels = padded.clone()
        special = {self.cls_id, self.eos_id, self.pad_id}
        prob = torch.full(labels.shape, self.mlm_prob)
        special_mask = torch.tensor([[t.item() in special for t in row] for row in padded])
        prob.masked_fill_(special_mask, 0.0)
        masked = torch.bernoulli(prob).bool()
        labels[~masked] = -100
        # 80% -> <mask>
        repl = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked
        padded[repl] = self.mask_id
        # 10% -> random AA
        rand = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked & ~repl
        random_ids = torch.randint(4, 29, padded.shape, dtype=torch.long)
        padded[rand] = random_ids[rand]
        # 10% unchanged
        return {"input_ids": padded, "attention_mask": attention_mask, "labels": labels}


# ── Training ───────────────────────────────────────────────────────
def train_species(species, args):
    suffix = f"_{args.model_suffix}" if args.model_suffix else ""
    output_dir = os.path.join(args.output_base, f"esm2_{species}{suffix}_final")
    os.makedirs(output_dir, exist_ok=True)

    # Model + Tokenizer
    logger.info(f"[{species}] Loading model...")
    try:
        tokenizer = EsmTokenizer.from_pretrained(args.model_path)
        model = EsmForMaskedLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, local_files_only=True)
    logger.info(f"  Model: {type(model).__name__}, "
                f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Data
    jsonl_path = os.path.join(args.data_dir, f"{species}_proteins.jsonl")
    dataset = ProteinDataset(jsonl_path, min_len=10, max_len=1024)

    val_size = min(int(len(dataset) * 0.005), 2000)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed))

    collator = MLMCollator(tokenizer, mlm_prob=0.15)

    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", 1)))
    global_batch = args.batch_size * args.grad_accum * world_size
    total_steps = (train_size // global_batch) * args.epochs
    warmup_steps = int(total_steps * 0.06)

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "tmp_trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=5e-5, weight_decay=0.01,
        warmup_steps=warmup_steps, lr_scheduler_type="linear",
        bf16=True, fp16=False,
        logging_steps=100, report_to=[],
        eval_strategy="steps", eval_steps=max(total_steps // 5, 1),
        save_strategy="no",  # don't save checkpoints during training
        dataloader_num_workers=4, dataloader_pin_memory=True,
        ddp_find_unused_parameters=True,
        seed=args.seed, run_name=f"esm2_{species}",
        disable_tqdm=False, remove_unused_columns=True,
    )

    logger.info(f"[{species}] GPUs={world_size}, global_batch={global_batch}, "
                f"total_steps={total_steps}, warmup={warmup_steps}")

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, tokenizer=tokenizer,
    )

    logger.info(f"[{species}] Starting training...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # Save ONLY final model
    logger.info(f"[{species}] Saving final model to {output_dir} ...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Clean trainer tmp
    import shutil
    shutil.rmtree(os.path.join(output_dir, "tmp_trainer"), ignore_errors=True)

    metrics = {
        "species": species, "train_samples": train_size, "val_samples": val_size,
        "total_steps": trainer.state.global_step, "training_time_min": round(elapsed/60, 1),
    }
    if trainer.state.best_metric:
        metrics["best_eval_loss"] = float(trainer.state.best_metric)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"[{species}] Done in {elapsed/60:.1f} min. Model: {output_dir}")
    return output_dir


# ── CLI ────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--species", required=True, choices=["human","rice","chlamydomonas"])
    p.add_argument("--model_path", required=True,
                   help="Path to base ESM2-650M model (e.g., facebook/esm2_t33_650M_UR50D)")
    p.add_argument("--data_dir", required=True,
                   help="Directory containing {species}_proteins.jsonl files")
    p.add_argument("--output_base", required=True,
                   help="Directory to save fine-tuned models")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--model_suffix", type=str, default="", help="Optional suffix for model dir name")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    train_species(args.species, args)


if __name__ == "__main__":
    main()
