#!/bin/bash
# =============================================================================
# Balanced Per-Species ESM2 MLM Fine-tuning — 4-GPU DDP Launch Script
# =============================================================================
# Adjusts rice and chlamydomonas training epochs to compensate for data imbalance:
#   - human:  224,457 seqs × 3 epochs  = ~670K samples (already trained, skip)
#   - rice:    41,787 seqs × 16 epochs = ~665K samples (~ human)
#   - chlamy:  15,277 seqs × 44 epochs = ~669K samples (~ human)
#
# Usage:
#   MODEL_PATH=/path/to/esm2_t33_650M_UR50D \
#   DATA_DIR=data/training \
#   OUTPUT_BASE=/path/to/output_models \
#   bash scripts/run_train_balanced.sh
#
#   nohup bash scripts/run_train_balanced.sh &> logs/train_balanced.log &
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_balanced_${TIMESTAMP}.log"

ACCELERATE_CONFIG="${SCRIPT_DIR}/accelerate_config.yaml"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_per_species.py"

# ── Configurable paths ───────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-facebook/esm2_t33_650M_UR50D}"
DATA_DIR="${DATA_DIR:-data/training}"
OUTPUT_BASE="${OUTPUT_BASE:-output_models}"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "4")

# ── Per-species epoch config ─────────────────────────────────────────
# human: not retrained (3 epochs, already exists)
# rice: 16 epochs -> 41,578 × 16 ≈ 665K
# chlamy: 44 epochs -> 15,201 × 44 ≈ 669K
get_epochs() {
    case "$1" in
        rice)           echo 16 ;;
        chlamydomonas)  echo 44 ;;
        *)              echo 3 ;;
    esac
}

MODEL_SUFFIX="balanced"

{
echo "================================================================"
echo "Balanced Per-Species Training"
echo "================================================================"
echo "Started: $(date)"
echo "GPUs:    ${NUM_GPUS}"
echo "Log:     ${LOG_FILE}"
echo "Model:   ${MODEL_PATH}"
echo "Data:    ${DATA_DIR}"
echo "Output:  ${OUTPUT_BASE}"
echo ""
echo "Epoch adjustment rationale:"
echo "  human:       224,457 seqs × 3  epochs = ~670K samples (baseline)"
echo "  rice:         41,787 seqs × 16 epochs = ~665K samples"
echo "  chlamy:       15,277 seqs × 44 epochs = ~669K samples"
echo ""

echo "--- Env Check ---"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}, GPUs {torch.cuda.device_count()}')" 2>&1
python3 -c "import transformers; print(f'Transformers {transformers.__version__}')" 2>&1
python3 -c "import accelerate; print(f'Accelerate {accelerate.__version__}')" 2>&1
for p in "${MODEL_PATH}" "${DATA_DIR}" "${TRAIN_SCRIPT}" "${ACCELERATE_CONFIG}"; do
    [ -e "$p" ] && echo "  ✓ $p" || echo "  ✗ MISSING: $p"
done
echo ""

for sp in human rice chlamydomonas; do
    lines=$(wc -l < "${DATA_DIR}/${sp}_proteins.jsonl")
    echo "  ${sp}: ${lines} raw sequences"
done
echo ""
echo "Models will be saved to: ${OUTPUT_BASE}/esm2_{species}_${MODEL_SUFFIX}_final/"
echo ""

echo "Starting in 5s..."
sleep 5

# ── Train rice + chlamydomonas (skip human) ──────────────────────────
for SPECIES in rice chlamydomonas; do
    EPOCH=$(get_epochs "$SPECIES")
    echo ""
    echo "================================================================"
    echo "  TRAINING: ${SPECIES} — ${EPOCH} epochs"
    echo "================================================================"
    echo "Started: $(date)"

    accelerate launch \
        --config_file "${ACCELERATE_CONFIG}" \
        --num_processes ${NUM_GPUS} \
        "${TRAIN_SCRIPT}" \
        --species "${SPECIES}" \
        --model_path "${MODEL_PATH}" \
        --data_dir "${DATA_DIR}" \
        --output_base "${OUTPUT_BASE}" \
        --batch_size 16 \
        --grad_accum 1 \
        --epochs ${EPOCH} \
        --model_suffix "${MODEL_SUFFIX}"

    RET=$?
    if [ ${RET} -ne 0 ]; then
        echo "ERROR: ${SPECIES} training failed (exit ${RET})"
        exit ${RET}
    fi

    echo "Finished: $(date)"
    echo "Model saved: ${OUTPUT_BASE}/esm2_${SPECIES}_${MODEL_SUFFIX}_final/"
    ls -lh "${OUTPUT_BASE}/esm2_${SPECIES}_${MODEL_SUFFIX}_final/" | head -10
done

echo ""
echo "================================================================"
echo "ALL DONE — 2 balanced models trained"
echo "================================================================"
echo "Models:"
for sp in rice chlamydomonas; do
    echo "  ${OUTPUT_BASE}/esm2_${sp}_${MODEL_SUFFIX}_final/"
done
echo "  (human: ${OUTPUT_BASE}/esm2_human_final/ — reused from previous 3-epoch training)"
echo ""
echo "Log: ${LOG_FILE}"
echo "Finished: $(date)"

} 2>&1 | tee "${LOG_FILE}"
