#!/bin/bash
# =============================================================================
# 5-Model polyQ Fitness Comparison — 1-GPU Launch Script
#   (base + merged + human + rice + chlamydomonas)
# =============================================================================
# Usage:
#   MODEL_DIR=/path/to/models \
#   QUERY_FILE=data/query/HTT_72Q.json \
#   OUTPUT_DIR=results \
#   bash scripts/run_fitness_balanced.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PROJECT_DIR}/logs/fitness_balanced_${TIMESTAMP}.log"

# ── Configurable paths ───────────────────────────────────────────────
MODEL_DIR="${MODEL_DIR:-models}"
QUERY_FILE="${QUERY_FILE:-data/query/HTT_72Q.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_DIR}"

# ── Env check ──
echo "================================================================"
echo "5-Model polyQ Fitness Comparison"
echo "================================================================"
echo "Started: $(date)"
echo ""

echo "--- GPU ---"
nvidia-smi 2>/dev/null | head -15 || true
echo ""

echo "--- Check models ---"
ALL_OK=true
for m in base merged human rice chlamydomonas; do
    p="${MODEL_DIR}/${m}"
    if [ -d "$p" ]; then
        echo "  ✓ $m"
    else
        echo "  ✗ $m: MISSING $p"
        ALL_OK=false
    fi
done
if [ "$ALL_OK" = false ]; then
    echo ""
    echo "WARNING: Some models missing."
    echo "Make sure training has completed before running this."
    echo "Missing models will be skipped during scoring."
fi

echo ""
echo "Output: ${OUTPUT_DIR}/"
echo "Log:    ${LOG_FILE}"
echo "Starting in 3s..."
sleep 3

# ── Launch ──
python3 -u scripts/zero_shot_fitness_balanced.py \
    --model_dir "${MODEL_DIR}" \
    --query_file "${QUERY_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    > "${LOG_FILE}" 2>&1

echo "Finished: $(date)"
echo "Log: ${LOG_FILE}"
