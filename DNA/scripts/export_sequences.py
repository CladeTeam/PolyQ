#!/usr/bin/env python3
"""Export the Q-sweep DNA sequences used for scoring."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from htt_delta.fasta import write_fasta
from htt_delta.sequences import SEQUENCE_BUILDERS, q_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, choices=sorted(SEQUENCE_BUILDERS))
    parser.add_argument("--output-fasta", required=True, type=Path)
    parser.add_argument("--output-metadata", required=True, type=Path)
    parser.add_argument("--q-min", type=int, default=10)
    parser.add_argument("--q-max", type=int, default=72)
    args = parser.parse_args()

    records = q_sweep(args.sequence, q_min=args.q_min, q_max=args.q_max)
    write_fasta(args.output_fasta, [(record.sequence_id, record.sequence) for record in records])

    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sequence_id", "sequence_family", "q_count", "dna_length", "cag_count", "caa_count"]
    with args.output_metadata.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({"sequence_id": record.sequence_id, **record.metadata})

    print(f"Wrote {args.output_fasta}")
    print(f"Wrote {args.output_metadata}")


if __name__ == "__main__":
    main()

