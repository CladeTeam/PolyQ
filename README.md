# PolyQ — Multi-Modal Sequence Analysis

This repository contains code, data, and analysis pipelines for polyQ-related
sequence studies across two modalities:

| Directory | Content | Status |
|-----------|---------|--------|
| `AA_Sequence/` | Protein-level polyQ scoring using species-adapted ESM2-650M | ✅ Ready |
| `DNA/` | DNA CDS-level analysis with CENO models | ✅ Models ready |

See each subdirectory for details.

## AA_Sequence — Species-Adapted polyQ Fitness Prediction

The ESM2-650M base model is fine-tuned on individual species proteomes
(*Chlamydomonas reinhardtii*, *Oryza sativa*, *Homo sapiens*) via masked
language modeling (MLM), producing species-adapted variants. These models
score polyQ-containing sequences, measuring how well each sequence fits the
endogenous proteome landscape of each species.

**Key result**: Species-adapted models exhibit maximum divergence from the
base ESM2 model in the intermediate polyQ range Q≈20–32.

See [`AA_Sequence/README.md`](AA_Sequence/README.md) for details on setup, training, and inference.

## DNA — CENO CDS-Level Sequence Analysis

CENO (Codon-Enhanced Nucleotide Optimizer) models for DNA coding sequence (CDS)
analysis, fine-tuned on *Chlamydomonas reinhardtii* and *Oryza sativa* CDS data.

| Model | HF Repo |
|-------|---------|
| CENO-chlamydomonas-cds | `CladeTeam/CENO-chlamydomonas-cds` |
| CENO-rice-cds | `CladeTeam/CENO-rice-cds` |

```bash
# Download DNA models
huggingface-cli download CladeTeam/CENO-rice-cds \
    --local-dir models/dna/ceno-rice-cds \
    --local-dir-use-symlinks False
huggingface-cli download CladeTeam/CENO-chlamydomonas-cds \
    --local-dir models/dna/ceno-chlamydomonas-cds \
    --local-dir-use-symlinks False
```

See [`DNA/`](DNA/) for detailed documentation.

**HuggingFace Collection** (all models): https://huggingface.co/collections/CladeTeam/polyq

## Reference

Lin, Z. *et al.* Evolutionary-scale prediction of atomic-level protein
structure with a language model. *Science* **379**, 1123–1130 (2023).
