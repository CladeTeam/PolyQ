#!/usr/bin/env bash
set -euo pipefail

python scripts/run_from_config.py \
  --config configs/rice_human_htt_72q_natural_trunc.json \
  --gpu 0

