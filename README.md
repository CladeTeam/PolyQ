# PolyQ — Multi-Modal Sequence Analysis

This repository contains code, data, and analysis pipelines for polyQ-related
sequence studies across two modalities:

| Directory | Content | Status |
|-----------|---------|--------|
| `AA_Sequence/` | Protein-level polyQ scoring using species-adapted ESM2-650M | ✅ Ready |
| `DNA/` | DNA-level polyQ scoring using the CENO DNA language model (Nemotron-H-derived) | ✅ Ready |

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

## DNA — CENO DNA Language Model polyQ Delta Scoring

The CENO DNA language model (1.3B parameters, derived from NVIDIA's Nemotron-H
hybrid Mamba/Transformer/MoE architecture) is fine-tuned on species-specific
coding sequences (CDS) from *Chlamydomonas reinhardtii*, *Oryza sativa*, and
*Homo sapiens*. The fine-tuned models score HTT exon1 polyQ repeat variants
(Q=10–72) by average per-token log-likelihood.

The main reported value is the **delta** between fine-tuned and pretrained
model likelihoods:

```text
delta(Q) = avg_log_likelihood(finetuned_model, sequence_Q)
         - avg_log_likelihood(pretrained_model, sequence_Q)
```

This quantifies how species-specific CDS fine-tuning shifts the model's
likelihood of polyQ sequences as the glutamine repeat expands.

**Models** (published on Hugging Face under `CladeTeam`):

| Model | repo_id |
|-------|---------|
| Base (preview) | `CladeTeam/CENO-base-1b-preview` |
| Chlamydomonas CDS finetune | `CladeTeam/CENO-chlamydomonas-cds` |
| Rice CDS finetune | `CladeTeam/CENO-rice-cds` |

The `DNA/` directory contains a minimal, auditable pipeline for CDS data
preparation, fine-tuning, Q-sweep scoring, and plotting, with three documented
HTT/polyQ sequence families and reference delta curves.

See [`DNA/README.md`](DNA/README.md) for details on setup, training, and scoring.

## Reference

Lin, Z. *et al.* Evolutionary-scale prediction of atomic-level protein
structure with a language model. *Science* **379**, 1123–1130 (2023).
