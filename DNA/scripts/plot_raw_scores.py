#!/usr/bin/env python3
"""Plot raw pretrained and finetuned avg log-likelihood curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    q_counts = []
    pretrained = []
    finetuned = []
    with args.input.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            q_counts.append(int(row["q_count"]))
            pretrained.append(float(row["pretrained_avg_log_likelihood"]))
            finetuned.append(float(row["finetuned_avg_log_likelihood"]))

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(q_counts, pretrained, color="#808080", linewidth=2.3, marker="o", markersize=3.5, label="Pretrained")
    ax.plot(q_counts, finetuned, color="#4daf4a", linewidth=2.5, marker="o", markersize=3.5, label="Finetuned")
    ax.set_title(args.title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Q count")
    ax.set_ylabel("Average log-likelihood")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

