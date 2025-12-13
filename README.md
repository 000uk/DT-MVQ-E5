python main.py --config experiments/phase1_e5_lora/config.yaml

# Dual-Target Strong KD for Book Embedding Reconstruction

Reconstructing domain-specific book embedding spaces via
strong knowledge distillation and multi-vector query attention
under metadata-free constraints.

## Motivation

General-purpose embedding models (e.g., E5, GTE) are optimized for
generic sentence similarity, but fail to simultaneously satisfy:

- Coarse-grained category separability (genre-level clustering)
- Fine-grained semantic similarity (content-level retrieval)

This limitation becomes critical in book search systems where
**explicit metadata is unavailable**, and all structural information
must be inferred solely from raw text (title + description).

## Key Challenges

1. Representation collapse under supervised contrastive learning
2. Catastrophic forgetting of the pretrained semantic manifold
3. Loss-scale imbalance between contrastive and distillation objectives
4. Information bottleneck of single-vector embeddings

## Core Contributions

- **Strong Knowledge Distillation**
  - Identified loss-scale mismatch (≈2000×) as the root cause
  - Restored gradient balance via large-scale KD (λ = 400–600)

- **Gradient Dynamics Analysis**
  - Empirically demonstrated gradient conflict between
    category clustering and semantic retrieval objectives

- **Multi-Vector Query Architecture**
  - Introduced latent query attention to disentangle
    genre-level and content-level representations

- **Dual-Target Strong KD**
  - Prevented asymmetric collapse by anchoring both vectors
    to the teacher semantic manifold

## Architecture Overview

We replace mean pooling with two learnable latent queries
(genre / content) sharing the same encoder backbone.

Both vectors are jointly optimized under:
- Contrastive loss (genre clustering)
- Dual-target knowledge distillation (semantic preservation)

## Training Strategy

- LoRA fine-tuning (r=16) to avoid full-model overfitting
- Strong KD with gradient-norm alignment
- High-momentum learning rate (1e-3) to escape KD-induced basins
- Sigmoid-based curriculum scheduling for genre/content balance

## Experimental Results

| Model | MRR |
|------|-----|
| Original E5 | 0.583 |
| Phase 1 (Strong KD) | 0.676 |
| Phase 2 (Proposed) | **0.706** |
| Oracle (with category) | 0.716 |

## Documentation

- 📄 [Technical Report](docs/technical_report.md)
- 📊 [Experiment Logs](experiments/)
- 🧠 [Gradient Analysis](docs/figures/)