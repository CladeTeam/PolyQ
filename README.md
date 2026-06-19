# ESM2Fitness — Species-Adapted Zero-Shot polyQ Fitness Prediction (Balanced)

This repository contains code and data for **species-adapted polyQ fitness scoring**
using the ESM2-650M protein language model, as presented in the collaborative
study with Fudan University (FDU).

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

## Repository Structure

```
ESM2Fitness_balanced/
├── README.md                          # This file
├── data/
│   ├── training/                      # Proteome data for MLM fine-tuning
│   │   ├── human_proteins.jsonl.gz
│   │   ├── rice_proteins.jsonl.gz
│   │   ├── chlamydomonas_proteins.jsonl.gz
│   │   └── dataset_stats.json
│   ├── query/
│   │   └── HTT_72Q.json              # HTT exon1 with 72Q query sequence
│   └── results/                       # Balanced experiment results
│       ├── polyQ_summary_all_models.csv
│       ├── pivot_mean_logP_ref.csv
│       ├── polyQ_fitness_all_models.csv.gz
│       └── natural_pll/              # Natural protein baseline scores
├── scripts/                           # Training & inference scripts
│   ├── train_per_species.py           # Per-species MLM fine-tuning
│   ├── zero_shot_fitness_balanced.py  # Multi-model polyQ scoring
│   ├── score_natural_pll.py           # Natural protein logP scoring
│   ├── run_train_balanced.sh          # Training launch script
│   ├── run_fitness_balanced.sh        # Inference launch script
│   ├── accelerate_config.yaml         # DDP configuration (4-GPU)
│   └── env_check.sh                   # Environment verification
├── analysis/                          # Downstream analysis & visualization
│   ├── generate_report_balanced.py    # Full report + all figures
│   ├── plot_delta_detail_clean.py     # Standalone delta-logP figure
│   ├── legend_method_bilingual.md     # Figure legends & methods (EN/ZH)
│   └── analysis_report.md            # Generated analysis report
└── figures/                           # Generated figures (PNG)
    ├── fig01_logP_vs_Q.png
    ├── fig01_delta_detail.png
    ├── fig02_region_boxplot.png
    ├── fig03_correlation.png
    ├── fig04_distribution.png
    ├── fig05_region_trends.png
    ├── fig06_delta_heatmap.png
    ├── fig07_high_logP_proteins.png
    ├── fig08_natural_polyq.png
    ├── fig09_natural_reference.png
    └── fig_delta_clean_rice_chlamy.png
```

## Models

Five models are compared:

| # | Model | Training Data | Epochs | Total Samples |
|---|-------|--------------|--------|---------------|
| 1 | **Base** | — (pretrained ESM2-650M) | — | — |
| 2 | **Merged** | 3-species merged (281,521 seqs) | 3 | ~670K |
| 3 | **Human** | Human proteome (224,457 seqs) | 3 | ~670K |
| 4 | **Rice (balanced)** | Rice proteome (41,787 seqs) | 16 | ~665K |
| 5 | **Chlamy (balanced)** | Chlamy proteome (15,277 seqs) | 44 | ~669K |

Epochs for Rice and Chlamy are scaled to match ~670K total training samples
across species, ensuring comparable optimization.

## Quick Start

### 1. Environment Setup

```bash
# Check dependencies
bash scripts/env_check.sh

# Required: PyTorch, Transformers, Accelerate, pandas, numpy, scipy, seaborn
pip install torch transformers accelerate pandas numpy scipy seaborn matplotlib
```

### 2. Training (Per-Species MLM Fine-Tuning)

Requires 4 GPUs with DDP. Downloads the base ESM2-650M model from HuggingFace
(or use a local path).

```bash
# Set paths
export MODEL_PATH="facebook/esm2_t33_650M_UR50D"  # or local path
export DATA_DIR="data/training"
export OUTPUT_BASE="output_models"

# Launch training (trains rice + chlamydomonas with balanced epochs)
bash scripts/run_train_balanced.sh
```

To train a single species:
```bash
python scripts/train_per_species.py \
    --species chlamydomonas \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_base "$OUTPUT_BASE" \
    --epochs 44 \
    --model_suffix balanced
```

### 3. Inference (Zero-Shot polyQ Scoring)

Requires 1 GPU with sufficient memory.

```bash
# Organize models into a single directory
# model_dir/
#   ├── base/           # ESM2-650M base
#   ├── merged/         # 3-species merged
#   ├── human/          # Human fine-tuned
#   ├── rice_balanced/  # Rice 16-epoch
#   └── chlamydomonas_balanced/  # Chlamy 44-epoch

python scripts/zero_shot_fitness_balanced.py \
    --model_dir path/to/models \
    --query_file data/query/HTT_72Q.json \
    --output_dir results
```

### 4. Analysis & Reporting

```bash
python analysis/generate_report_balanced.py \
    --results_dir results \
    --natural_pll_dir results/natural_pll \
    --training_data_dir data/training \
    --output_dir analysis_output
```

## Scoring Metric

The species-adapted delta score:

$$\Delta\log P_{\text{species}} = \overline{\log P(\text{ref})}_{\text{species-adapted}} - \overline{\log P(\text{ref})}_{\text{base}}$$

A positive ΔlogP suggests the candidate sequence is more concordant with
that species' endogenous proteome landscape.

## Data Sources

- **Human proteome**: Ensembl 115, GRCh38
- **Rice proteome**: Ensembl Plants 62, IRGSP-1.0
- **Chlamy proteome**: Ensembl Plants 62, v5.5
- **Base model**: ESM2-t33-650M-UR50D (Meta AI)

## Reference

Lin, Z. *et al.* Evolutionary-scale prediction of atomic-level protein
structure with a language model. *Science* **379**, 1123–1130 (2023).

## License

This repository is shared for collaborative research purposes with Fudan University.
Please contact the authors before redistribution.
