#!/usr/bin/env python3
"""Plot q_count,delta CSV output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: Path) -> tuple[list[int], list[float]]:
    q_counts = []
    deltas = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            q_counts.append(int(row["q_count"]))
            deltas.append(float(row["delta"]))
    return q_counts, deltas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--label", default="Finetuned - pretrained")
    parser.add_argument("--color", default="#4daf4a")
    args = parser.parse_args()

    q_counts, deltas = read_csv(args.input)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(q_counts, deltas, color=args.color, linewidth=2.5, marker="o", markersize=3.5, label=args.label)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.2)
    ax.set_title(args.title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Q count")
    ax.set_ylabel("Delta avg log-likelihood")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

