#!/usr/bin/env bash
set -euo pipefail

python scripts/run_from_config.py \
  --config configs/chlamydomonas_cr_htt_72q_original.json \
  --gpu 0

