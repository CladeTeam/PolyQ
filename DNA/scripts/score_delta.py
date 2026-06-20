#!/usr/bin/env python3
"""Score pretrained and finetuned models on an HTT/polyQ Q sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from htt_delta.fasta import write_fasta
from htt_delta.scoring import choose_device, score_model
from htt_delta.sequences import SEQUENCE_BUILDERS, q_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, choices=sorted(SEQUENCE_BUILDERS))
    parser.add_argument("--pretrained-model", required=True)
    parser.add_argument("--finetuned-model", required=True)
    parser.add_argument("--model-code-dir", required=True)
    parser.add_argument("--output", required=True, type=Path, help="q_count,delta CSV")
    parser.add_argument("--details-output", required=True, type=Path)
    parser.add_argument("--sequence-output", required=True, type=Path)
    parser.add_argument("--q-min", type=int, default=10)
    parser.add_argument("--q-max", type=int, default=72)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = choose_device(args.gpu)
    records = q_sweep(args.sequence, q_min=args.q_min, q_max=args.q_max)
    print(f"Device: {device}")
    print(f"Sequence family: {args.sequence}")
    print(f"Records: {len(records)}")

    write_fasta(args.sequence_output, [(record.sequence_id, record.sequence) for record in records])
    pretrained_scores = score_model(
        args.pretrained_model, args.model_code_dir, records, device, "pretrained"
    )
    finetuned_scores = score_model(
        args.finetuned_model, args.model_code_dir, records, device, "finetuned"
    )

    rows = []
    for record, pretrained, finetuned in zip(records, pretrained_scores, finetuned_scores):
        rows.append({
            "sequence_id": record.sequence_id,
            **record.metadata,
            "pretrained_avg_log_likelihood": pretrained,
            "finetuned_avg_log_likelihood": finetuned,
            "delta": finetuned - pretrained,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["q_count", "delta"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"q_count": row["q_count"], "delta": row["delta"]})

    args.details_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence_id",
        "sequence_family",
        "q_count",
        "dna_length",
        "cag_count",
        "caa_count",
        "pretrained_avg_log_likelihood",
        "finetuned_avg_log_likelihood",
        "delta",
    ]
    with args.details_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})

    print(f"Wrote {args.output}")
    print(f"Wrote {args.details_output}")
    print(f"Wrote {args.sequence_output}")


if __name__ == "__main__":
    main()

