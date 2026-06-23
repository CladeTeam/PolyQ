# AA_Sequence — Species-Adapted polyQ Fitness Prediction

Code and data for **species-adapted polyQ fitness scoring** using the ESM2-650M
protein language model.

## Overview

The ESM2-650M base model is fine-tuned on individual species proteomes
(*Chlamydomonas reinhardtii*, *Oryza sativa*, *Homo sapiens*) via masked
language modeling (MLM), producing species-adapted variants. These models
are then used to score polyQ-containing sequences, measuring how well each
sequence fits the endogenous proteome landscape of each species.

**Key result**: Species-adapted models exhibit maximum divergence from the
base ESM2 model in the intermediate polyQ range Q≈20–32, identifying this
as a transition zone where species-specific proteome context most sensitively
modulates sequence likelihood.

## Directory Structure

```
AA_Sequence/
├── README.md
├── data/
│   ├── training/                      # Proteome data for MLM fine-tuning
│   ├── query/
│   │   └── HTT_72Q.json              # HTT exon1 with 72Q query sequence
│   └── results/                       # Experiment results
│       └── natural_pll/              # Natural protein baseline scores
├── scripts/                           # Training & inference scripts
├── analysis/                          # Downstream analysis & visualization
└── figures/                           # Generated figures (PNG)
```

## Models

| # | Model | Training Data | Epochs | Total Samples | Source |
|---|-------|--------------|--------|---------------|--------|
| 1 | **Base** | — (pretrained ESM2-650M) | — | — | Meta AI |
| 2 | **Merged** | 3-species merged (281,521 seqs) | 3 | ~670K | Train yourself |
| 3 | **Human** | Human proteome (224,457 seqs) | 3 | ~670K | Train yourself |
| 4 | **Rice** | Rice proteome (41,787 seqs) | 3 | ~125K | CladeTeam HF |
| 5 | **Chlamy** | Chlamy proteome (15,277 seqs) | 3 | ~46K | CladeTeam HF |

All species are trained for 3 epochs each (standard MLM fine-tuning).

---

## Model Setup

The inference scripts expect models under a single directory with the following
layout (set via `MODEL_DIR` / `--model_dir`):

```
models/
├── base/                          # facebook/esm2_t33_650M_UR50D
├── merged/                        # 3-species merged (train yourself)
├── human/                         # Human proteome (train yourself)
├── rice/                          # CladeTeam/polyq-esm2-rice
├── chlamydomonas/                 # CladeTeam/polyq-esm2-chlamy
```

### Step 1: Base ESM2-650M

Download from Meta AI's HuggingFace repository:

```bash
pip install huggingface_hub
huggingface-cli download facebook/esm2_t33_650M_UR50D \
    --local-dir models/base \
    --local-dir-use-symlinks False
```

Alternatively, let HuggingFace Transformers cache it automatically on first use
by passing the model ID directly to the training script.

### Step 2: Fine-tuned Species Models (ours)

Download from the CladeTeam collection on HuggingFace:

```bash
# Rice (3 epochs)
huggingface-cli download CladeTeam/polyq-esm2-rice \
    --local-dir models/rice \
    --local-dir-use-symlinks False

# Chlamydomonas (3 epochs)
huggingface-cli download CladeTeam/polyq-esm2-chlamy \
    --local-dir models/chlamydomonas \
    --local-dir-use-symlinks False
```

**HuggingFace Collection**: https://huggingface.co/collections/CladeTeam/polyq

### Step 3: Human & Merged Models

These are not distributed as weights — train them locally if needed:

```bash
export MODEL_PATH="models/base"          # or "facebook/esm2_t33_650M_UR50D"
export DATA_DIR="data/training"
export OUTPUT_BASE="output_models"

# Train all three species (3 epochs each)
bash scripts/run_train.sh

# Then symlink/copy into the shared model directory:
ln -s "$(pwd)/output_models/esm2_human_final" models/human
ln -s "$(pwd)/output_models/esm2_merged_final" models/merged
```

> **Note**: The polyQ scoring script gracefully skips missing models, so you can
> run inference with only `base`, `rice`, and `chlamydomonas`.
> Missing models are logged as warnings and excluded from the comparison.

### Step 4: Verify Layout

```bash
for m in base merged human rice chlamydomonas; do
    if [ -d "models/${m}" ]; then
        echo "✓ models/${m}"
    else
        echo "✗ models/${m} — SKIPPED"
    fi
done
```

---

## Quick Start

### Environment

```bash
bash scripts/env_check.sh
pip install torch transformers accelerate pandas numpy scipy seaborn matplotlib
```

### Training (4-GPU DDP)

```bash
export MODEL_PATH="facebook/esm2_t33_650M_UR50D"
export DATA_DIR="data/training"
export OUTPUT_BASE="output_models"
bash scripts/run_train.sh
```

### Inference (1 GPU)

```bash
export MODEL_DIR="models"
export QUERY_FILE="data/query/HTT_72Q.json"
export OUTPUT_DIR="results"
bash scripts/run_fitness_balanced.sh
```

### Analysis

```bash
python analysis/generate_report_balanced.py \
    --results_dir results \
    --natural_pll_dir results/natural_pll \
    --training_data_dir data/training \
    --output_dir analysis_output
```

## Scoring Metric

$$\Delta\log P_{\text{species}} = \overline{\log P(\text{ref})}_{\text{species-adapted}} - \overline{\log P(\text{ref})}_{\text{base}}$$

## Data Sources

- Human proteome: Ensembl 115, GRCh38
- Rice proteome: Ensembl Plants 62, IRGSP-1.0
- Chlamy proteome: Ensembl Plants 62, v5.5
- Base model: ESM2-t33-650M-UR50D (Meta AI)

## Reference

Lin, Z. *et al.* Evolutionary-scale prediction of atomic-level protein
structure with a language model. *Science* **379**, 1123–1130 (2023).
