#!/usr/bin/env python3
"""
Score logP(ref) for natural proteins using the base ESM2 model.
Samples proteins and scores a subset of positions for distribution comparison.

Usage:
    python scripts/score_natural_pll.py \
        --model_path <path_to_esm2_base> \
        --data_dir data/training \
        --output_dir results/natural_pll
"""

import json, os, random, time, argparse
import numpy as np
import torch
from transformers import EsmForMaskedLM, EsmTokenizer


def load_model(device, model_path):
    tok = EsmTokenizer.from_pretrained(model_path)
    model = EsmForMaskedLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, local_files_only=True)
    model = model.to(device).eval()
    a2i = {}
    for t, tid in tok.get_vocab().items():
        if len(t) == 1 and t.isalpha() and t in "LAGVSERTIDPKQNFYMHWC":
            a2i[t] = tid
    return model, tok, a2i


def load_sequences(species, data_dir, min_len=50, max_len=800):
    fpath = os.path.join(data_dir, f"{species}_proteins.jsonl")
    seqs = []
    with open(fpath) as f:
        for line in f:
            rec = json.loads(line.strip())
            seq = rec.get("seq", "")
            if min_len <= len(seq) <= max_len:
                seqs.append(seq)
    return seqs


@torch.no_grad()
def score_positions(model, tok, a2i, sequences, device, n_proteins=500, n_pos=10, seed=42):
    rng = random.Random(seed)
    sampled = rng.sample(sequences, min(n_proteins, len(sequences)))
    mask_id = tok.mask_token_id; cls_id = tok.cls_token_id; eos_id = tok.eos_token_id
    rows = []
    for pi, seq in enumerate(sampled):
        ids = [cls_id] + [a2i.get(aa, tok.unk_token_id) for aa in seq] + [eos_id]
        ids = torch.tensor(ids, dtype=torch.long)
        valid = list(range(1, len(ids) - 1))
        chosen = sorted(rng.sample(valid, min(n_pos, len(valid))))
        for pos in chosen:
            masked = ids.clone()
            masked[pos] = mask_id
            logits = model(masked.unsqueeze(0).to(device)).logits[0, pos].float()
            lp = torch.log_softmax(logits, dim=-1)
            pll = lp[ids[pos].item()].item()
            # Get AA char from id
            wt_id = ids[pos].item()
            wt_aa = "?"
            for aa, aid in a2i.items():
                if aid == wt_id: wt_aa = aa; break
            rows.append({"protein_idx": pi, "position": pos, "wildtype": wt_aa, "pll": pll})
        if (pi+1) % 100 == 0:
            print(f"  {pi+1}/{len(sampled)} proteins done")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to base ESM2 model")
    parser.add_argument("--data_dir", default="data/training",
                        help="Directory with {species}_proteins.jsonl files")
    parser.add_argument("--output_dir", default="results/natural_pll",
                        help="Directory to save output CSVs")
    parser.add_argument("--species", nargs="+", default=["human", "chlamydomonas"],
                        help="Species to score")
    parser.add_argument("--n_proteins", type=int, default=500,
                        help="Number of proteins to sample per species")
    parser.add_argument("--n_pos", type=int, default=10,
                        help="Number of positions to score per protein")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, tok, a2i = load_model(device, args.model_path)

    for species in args.species:
        print(f"\n=== {species} ===")
        seqs = load_sequences(species, args.data_dir)
        print(f"  Loaded {len(seqs):,} sequences (50-800 aa)")
        t0 = time.time()
        rows = score_positions(model, tok, a2i, seqs, device,
                               n_proteins=args.n_proteins, n_pos=args.n_pos)
        elapsed = time.time() - t0
        print(f"  Scored {len(rows)} positions in {elapsed:.1f}s")

        import pandas as pd
        df = pd.DataFrame(rows)
        out_path = os.path.join(args.output_dir, f"{species}_natural_pll.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
