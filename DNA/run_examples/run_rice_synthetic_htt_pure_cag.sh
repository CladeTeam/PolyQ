#!/usr/bin/env bash
set -euo pipefail

python scripts/run_from_config.py \
  --config configs/rice_synthetic_htt_pure_cag.json \
  --gpu 0

