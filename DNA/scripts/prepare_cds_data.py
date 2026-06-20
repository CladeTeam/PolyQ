#!/usr/bin/env python3
"""Prepare CDS FASTA files for finetuning."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from htt_delta.fasta import read_fasta, write_fasta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cds", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-frac", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-len", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=8192)
    args = parser.parse_args()

    records = read_fasta(args.input_cds, min_len=args.min_len, max_len=args.max_len)
    random.seed(args.seed)
    random.shuffle(records)
    split = int(len(records) * args.train_frac)

    write_fasta(args.output_dir / "train.fa", records[:split])
    write_fasta(args.output_dir / "val.fa", records[split:])

    print(f"Input records kept: {len(records)}")
    print(f"Train records: {split}")
    print(f"Val records: {len(records) - split}")


if __name__ == "__main__":
    main()

