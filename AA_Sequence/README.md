# AA_Sequence — Species-Adapted polyQ Fitness Prediction (Balanced)

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
│   └── results/                       # Balanced experiment results
│       └── natural_pll/              # Natural protein baseline scores
├── scripts/                           # Training & inference scripts
├── analysis/                          # Downstream analysis & visualization
└── figures/                           # Generated figures (PNG)
```

## Models

| # | Model | Training Data | Epochs | Total Samples |
|---|-------|--------------|--------|---------------|
| 1 | **Base** | — (pretrained ESM2-650M) | — | — |
| 2 | **Merged** | 3-species merged (281,521 seqs) | 3 | ~670K |
| 3 | **Human** | Human proteome (224,457 seqs) | 3 | ~670K |
| 4 | **Rice (balanced)** | Rice proteome (41,787 seqs) | 16 | ~665K |
| 5 | **Chlamy (balanced)** | Chlamy proteome (15,277 seqs) | 44 | ~669K |

Epochs for Rice and Chlamy are scaled to match ~670K total training samples.

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
bash scripts/run_train_balanced.sh
```

### Inference (1 GPU)

```bash
python scripts/zero_shot_fitness_balanced.py \
    --model_dir path/to/models \
    --query_file data/query/HTT_72Q.json \
    --output_dir results
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
