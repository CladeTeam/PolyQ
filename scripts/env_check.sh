#!/bin/bash
# =============================================================================
# Environment Check — verify all dependencies are ready
# =============================================================================
set -euo pipefail

echo "================================================================"
echo "ESM2 Fitness — Environment Check"
echo "================================================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

echo "--- Python ---"
python3 --version 2>&1
which python3
echo ""

echo "--- CUDA / GPU ---"
nvidia-smi 2>/dev/null | head -15 || echo "  No NVIDIA GPU / nvidia-smi not found"
echo ""

echo "--- PyTorch ---"
python3 -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>&1
echo ""

echo "--- Transformers ---"
python3 -c "import transformers; print(f'  Transformers {transformers.__version__}')" 2>&1
echo ""

echo "--- Accelerate ---"
python3 -c "
import accelerate
print(f'  Accelerate {accelerate.__version__}')
print(f'  Default device: {accelerate.state.default_device}')
" 2>&1
echo ""

echo "--- Other Dependencies ---"
for pkg in numpy pandas scipy seaborn matplotlib; do
    python3 -c "import $pkg; v=getattr($pkg,'__version__','?'); print(f'  {pkg}: {v}')" 2>&1
done
echo ""

echo "--- Check Accelerate Config ---"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/accelerate_config.yaml" ]; then
    echo "  ✓ accelerate_config.yaml found"
    cat "${SCRIPT_DIR}/accelerate_config.yaml"
else
    echo "  ✗ accelerate_config.yaml MISSING"
fi
echo ""

echo "================================================================"
echo "Environment check complete."
echo "================================================================"
