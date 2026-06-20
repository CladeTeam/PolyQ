#!/usr/bin/env python3
"""Run scoring and plotting from a JSON config file."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    with args.config.open() as handle:
        config = json.load(handle)

    root = Path(__file__).resolve().parents[1]
    outputs = config["outputs"]

    run([
        sys.executable,
        str(root / "scripts" / "score_delta.py"),
        "--sequence",
        config["sequence"],
        "--pretrained-model",
        config["pretrained_model"],
        "--finetuned-model",
        config["finetuned_model"],
        "--model-code-dir",
        config["model_code_dir"],
        "--output",
        outputs["delta_csv"],
        "--details-output",
        outputs["details_csv"],
        "--sequence-output",
        outputs["sequences_fasta"],
        "--q-min",
        str(config.get("q_min", 10)),
        "--q-max",
        str(config.get("q_max", 72)),
        "--gpu",
        str(args.gpu),
    ])

    run([
        sys.executable,
        str(root / "scripts" / "plot_delta.py"),
        "--input",
        outputs["delta_csv"],
        "--output",
        outputs["delta_png"],
        "--title",
        config["description"],
    ])

    run([
        sys.executable,
        str(root / "scripts" / "plot_raw_scores.py"),
        "--input",
        outputs["details_csv"],
        "--output",
        outputs["raw_png"],
        "--title",
        config["description"] + " raw scores",
    ])


if __name__ == "__main__":
    main()

