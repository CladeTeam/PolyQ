"""Scoring utilities for causal DNA language models."""

from __future__ import annotations

import gc

import torch
from tqdm import tqdm

from .model_io import load_model_and_tokenizer
from .sequences import SequenceRecord


def choose_device(gpu: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def average_log_likelihood(model, tokenizer, dna_sequence: str, device: torch.device) -> float:
    input_ids = tokenizer.encode(dna_sequence)
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids_tensor).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    target_ids = input_ids_tensor[0, 1:]
    predicted_log_probs = log_probs[0, :-1, :]
    token_log_probs = predicted_log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    return float(token_log_probs.mean().item())


def score_model(
    model_path: str,
    model_code_dir: str,
    records: list[SequenceRecord],
    device: torch.device,
    label: str,
) -> list[float]:
    model, tokenizer, _ = load_model_and_tokenizer(model_path, model_code_dir, device)
    scores = [
        average_log_likelihood(model, tokenizer, record.sequence, device)
        for record in tqdm(records, desc=f"Scoring {label}")
    ]
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores

