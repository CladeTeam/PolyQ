#!/usr/bin/env python3
"""
Zero-Shot Fitness — 5-Model Comparison
(base + merged + human + rice + chlamydomonas)

Species-adapted ESM2 models fine-tuned on individual proteomes (3 epochs each).
Scores polyQ sequences against all 5 models, outputting per-position and
per-sequence fitness metrics.

Output to results/ directory.

Usage (1 GPU):
    python scripts/zero_shot_fitness_balanced.py \
        --model_dir <path_to_models> \
        --query_file data/query/HTT_72Q.json \
        --output_dir results/
"""

import argparse, json, logging, os, time
import numpy as np
import torch
from transformers import EsmForMaskedLM, EsmTokenizer, AutoModelForMaskedLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────
def parse_htt(query_file):
    with open(query_file) as f:
        data = json.load(f)
    full_seq = data["protein_sequence"]
    nterm = full_seq[:data["polyQ_start"] - 1]
    cterm = full_seq[data["polyQ_end"]:]
    logger.info(f"HTT: N-term={len(nterm)}, C-term={len(cterm)}")
    return nterm, cterm


def load_model(path, device):
    logger.info(f"Loading: {path}")
    try:
        tok = EsmTokenizer.from_pretrained(path)
        model = EsmForMaskedLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForMaskedLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                      local_files_only=True)
    model = model.to(device).eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"  {type(model).__name__}, {n:.1f}M params")
    return model, tok


def make_aa_map(tok):
    standard = list("LAGVSERTIDPKQNFYMHWC")
    a2i, i2a = {}, {}
    for t, tid in tok.get_vocab().items():
        if t in standard:
            a2i[t] = tid; i2a[tid] = t
    return a2i, i2a


@torch.no_grad()
def score_all_positions(model, token_ids, a2i, i2a, mask_id, device, batch_size=32):
    """Score all valid positions, return list of {position, wildtype, scores, pll}."""
    valid = list(range(1, len(token_ids) - 1))
    results = []
    for bstart in range(0, len(valid), batch_size):
        bpos = valid[bstart:bstart + batch_size]
        bsize = len(bpos)
        inp = token_ids.unsqueeze(0).repeat(bsize, 1).to(device)
        for i, p in enumerate(bpos):
            inp[i, p] = mask_id
        logits = model(inp).logits
        for i, p in enumerate(bpos):
            wt_id = token_ids[p].item()
            wt_aa = i2a.get(wt_id, "?")
            if wt_aa == "?": continue
            lp = torch.log_softmax(logits[i, p].float(), dim=-1)
            pll = lp[wt_id].item()
            scores = {}
            for aa, aid in a2i.items():
                if aa != wt_aa:
                    scores[aa] = float(lp[aid].item() - lp[wt_id].item())
            results.append({"position": p, "wildtype": wt_aa, "scores": scores, "pll": pll})
    return results


# ── Main ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True,
                        help="Directory containing model subdirectories: base, merged, human, "
                             "rice, chlamydomonas")
    parser.add_argument("--query_file", default="data/query/HTT_72Q.json",
                        help="Path to HTT-72Q query JSON")
    parser.add_argument("--output_dir", default="results",
                        help="Directory to save output CSVs")
    parser.add_argument("--q_min", type=int, default=10)
    parser.add_argument("--q_max", type=int, default=72)
    parser.add_argument("--q_step", type=int, default=1)
    parser.add_argument("--score_batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    nterm, cterm = parse_htt(args.query_file)
    q_lengths = sorted(set(range(args.q_min, args.q_max + 1, args.q_step)) | {args.q_max})

    # ── Build model paths ──────────────────────────────────────────
    MODEL_NAMES = ["base", "merged", "human", "rice", "chlamydomonas"]
    MODEL_PATHS = {}
    for name in MODEL_NAMES:
        path = os.path.join(args.model_dir, name)
        if os.path.exists(path):
            MODEL_PATHS[name] = path
        else:
            logger.warning(f"Model directory not found: {path}, skipping {name}")

    # ── Load all models ──────────────────────────────────────────
    models = {}
    for name, path in MODEL_PATHS.items():
        m, tok = load_model(path, device)
        a2i, i2a = make_aa_map(tok)
        models[name] = (m, tok, a2i, i2a, tok.mask_token_id)

    logger.info(f"Loaded {len(models)} models: {list(models.keys())}")

    # ── Score polyQ sequences ──────────────────────────────────────
    all_rows = []  # for CSV
    summaries = []  # per-model, per-Q summary

    for ql in q_lengths:
        seq = nterm + "Q" * ql + cterm
        logger.info(f"\nQ={ql}: seq_len={len(seq)}")

        for mname, (model, tok, a2i, i2a, mask_id) in models.items():
            t0 = time.time()
            ids = torch.tensor(
                [tok.cls_token_id] +
                [a2i.get(aa, tok.unk_token_id) for aa in seq] +
                [tok.eos_token_id], dtype=torch.long)

            results = score_all_positions(model, ids, a2i, i2a, mask_id, device,
                                          batch_size=args.score_batch)
            elapsed = time.time() - t0

            # Aggregate
            all_scores_flat = []
            all_plls = []
            for r in results:
                all_scores_flat.extend(r["scores"].values())
                all_plls.append(r["pll"])
                for mut, sc in r["scores"].items():
                    all_rows.append({
                        "model": mname, "q_length": ql,
                        "position": r["position"], "wildtype": r["wildtype"],
                        "mutant": mut, "score": sc, "pll": r["pll"],
                    })

            mean_score = np.mean(all_scores_flat) if all_scores_flat else float("nan")
            mean_pll = np.mean(all_plls) if all_plls else float("nan")
            summaries.append({
                "model": mname, "q_length": ql, "seq_length": len(seq),
                "positions_scored": len(results),
                "mean_score": mean_score, "mean_pll": mean_pll,
                "total_score": np.sum(all_scores_flat) if all_scores_flat else float("nan"),
            })
            logger.info(f"  [{mname}] {len(results)} pos in {elapsed:.1f}s, "
                        f"mean_score={mean_score:.4f}, mean_pll={mean_pll:.4f}")

    # ── Save CSVs ──────────────────────────────────────────────────
    import pandas as pd
    # Per-position details
    df = pd.DataFrame(all_rows)
    detail_path = os.path.join(args.output_dir, "polyQ_fitness_all_models.csv")
    df.to_csv(detail_path, index=False)
    logger.info(f"Saved: polyQ_fitness_all_models.csv ({len(df)} rows)")

    # Summary
    df_sum = pd.DataFrame(summaries)
    summary_path = os.path.join(args.output_dir, "polyQ_summary_all_models.csv")
    df_sum.to_csv(summary_path, index=False)
    logger.info(f"Saved: polyQ_summary_all_models.csv ({len(df_sum)} rows)")

    # Pivot tables
    pivot_score = df_sum.pivot(index="q_length", columns="model", values="mean_score")
    pivot_score.to_csv(os.path.join(args.output_dir, "pivot_mean_score.csv"))
    pivot_pll = df_sum.pivot(index="q_length", columns="model", values="mean_pll")
    pivot_pll.to_csv(os.path.join(args.output_dir, "pivot_mean_pll.csv"))
    # Also save as mean_logP_ref for compatibility with generate_report.py
    pivot_pll.to_csv(os.path.join(args.output_dir, "pivot_mean_logP_ref.csv"))
    pivot_pll.to_csv(os.path.join(args.output_dir, "pivot_mean_logP.csv"))

    logger.info(f"\nDone! Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
