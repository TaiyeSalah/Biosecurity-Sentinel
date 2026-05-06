# Architecture

## System Overview

```
SMILES string
      │
      ▼
┌─────────────────────────────────┐
│   Structural Alert Screening    │  ← Deterministic. Runs first. Always.
│   (SMARTS vs CWC analogs)       │  ← CRITICAL alert → immediate HALT
└─────────────────────────────────┘
      │ (if no CRITICAL alert)
      ▼
┌─────────────────────────────────┐
│   3D Conformer Generation       │  ← RDKit ETKDGv3
│   Strategy 1: ETKDGv3 + MMFF94 │
│   Strategy 2: ETKDGv3 + random seed │
│   Strategy 3: Classic ETDG      │
│   Failure → flag CONFORMER_FAILED, never skip silently │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   Atomic Graph Construction     │  ← z (atomic numbers) + pos (3D coords)
│   (PyTorch Geometric Data)      │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   SchNet Encoder                │  ← 6 interaction layers, 10Å cutoff
│   (distance-aware message       │  ← Correctly distinguishes enantiomers
│    passing, 3D geometry)        │  ← Pretrained on QM9 (130k molecules)
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   Shared Dense Layers           │  ← 512 → 256, SiLU activation
│   (post-encoder)                │  ← LayerNorm
└─────────────────────────────────┘
      │
      ├──────────────────┬──────────────────┐
      ▼                  ▼                  ▼
┌──────────┐      ┌──────────┐      ┌──────────────┐
│ Toxicity │      │ Hepatic  │      │  Dual-Use    │
│  Head    │      │Clearance │      │  Risk Head   │
│(sigmoid) │      │(softplus)│      │  (sigmoid)   │
└──────────┘      └──────────┘      └──────────────┘
      │                  │                  │
      └──────────────────┴──────────────────┘
                         │
                         ▼
      ┌─────────────────────────────────────┐
      │   Monte Carlo Dropout Inference     │
      │   30 forward passes, dropout ON     │
      │   → mean + std + 95% CI per task    │
      │   std > 0.15 → flag UNKNOWN (OOD)   │
      └─────────────────────────────────────┘
                         │
                         ▼
      ┌─────────────────────────────────────┐
      │   Risk Classification               │
      │   + Clearance Mitigation Check      │
      │   + Recommendation Generation       │
      │   + JSONL Audit Log Entry           │
      └─────────────────────────────────────┘
```

## Why 3D matters

Standard 2D molecular fingerprint models (ECFP4, MACCS) produce identical
representations for R and S enantiomers — mirror-image molecules that can have
completely different toxicity profiles. Thalidomide is the canonical example:
the R-enantiomer was a safe sedative; the S-enantiomer caused severe birth defects.

SchNet's message passing operates on interatomic distances in 3D space. The same
pair of enantiomers produces distinct 3D graphs, distinct embeddings, and
potentially distinct toxicity predictions. This is not a marginal improvement —
it is the difference between correct and incorrect for a significant class of
biologically active compounds.

## Why MC Dropout matters for safety

A standard neural network gives a single point estimate. It cannot express
uncertainty. Trained on Tox21 (12,000 compounds), it will encounter molecules
far outside its training distribution — and it will return confident predictions
on them, because it has no mechanism to detect novelty.

Monte Carlo Dropout approximates Bayesian inference. By running 30 stochastic
forward passes (dropout active during inference), we get a distribution over
predictions. The standard deviation of that distribution is our uncertainty signal.

When std > 0.15, the molecule is flagged UNKNOWN. In biosecurity, a tool that
says "I don't know, test this experimentally" is far safer than a tool that says
"safe" with 0.97 confidence when it shouldn't.

## Training phases

**Phase 1 — QM9 pretraining (~12 hours, A100)**
Train SchNet encoder on quantum chemistry energy prediction (internal energy
at 0K) across 130,000 small organic molecules. This teaches the encoder to
read 3D molecular geometry before any toxicity signal is introduced.

**Phase 2 — Tox21 + ToxCast fine-tuning (~10 hours, A100)**
Fine-tune the full multi-task model on 12,000 labelled toxicity compounds
across 12 endpoints. Kendall uncertainty weighting prevents gradient
interference between the toxicity and clearance tasks.

## Structural alert library

See `docs/structural_alerts.md` for the full SMARTS library with sources.
All patterns are derived from CWC Schedule 1 analogs and published
biosecurity literature. The library is intentionally open and auditable.
