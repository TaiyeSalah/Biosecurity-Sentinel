"""
Biosecurity Sentinel & Threat Forecaster
SchNet 3D Graph Neural Network — model.py

Architecture:
  SMILES → RDKit 3D conformer → SchNet interaction blocks → Multi-task head
  Toxicity Score + Hepatic Clearance + MC Dropout Uncertainty

References:
  - Schütt et al. (2017) "SchNet: A continuous-filter convolutional neural network
    for modeling quantum interactions." NeurIPS.
  - Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation." ICML.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_add_pool


# ─── Configuration ────────────────────────────────────────────────────────────
@dataclass
class SchNetConfig:
    # SchNet interaction blocks
    n_interactions:       int   = 3        # number of interaction blocks
    n_filters:            int   = 128      # number of continuous filters
    n_gaussians:          int   = 64       # radial basis functions
    cutoff:               float = 10.0     # interaction cutoff (Angstrom)
    max_z:                int   = 100      # maximum atomic number

    # Molecular embedding
    embedding_dim:        int   = 64       # atom embedding dimension

    # Readout
    readout_dim:          int   = 256      # molecular fingerprint dimension

    # Multi-task head
    hidden_dim:           int   = 128
    head_hidden_dim:      int   = 64
    dropout_p:            float = 0.20    # MC Dropout probability

    # Uncertainty threshold
    ood_std_threshold:    float = 0.25    # flag as UNKNOWN if MC std > this

    # MC Dropout
    n_mc_samples:         int   = 50


# ─── Radial Basis Functions ───────────────────────────────────────────────────
class GaussianSmearing(nn.Module):
    """
    Expand pairwise distances into a basis of Gaussian functions.
    This is the key operation in SchNet: distances → continuous filter basis.

    d → [exp(-(d - mu_k)^2 / (2*sigma^2)) for k in 0..n_gaussians]

    The Gaussian centers mu_k are evenly spaced from 0 to cutoff.
    """
    def __init__(self, start: float = 0.0, stop: float = 10.0, n_gaussians: int = 64):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        self.register_buffer("offset", offset)
        self.coeff = -0.5 / ((stop - start) / n_gaussians) ** 2

    def forward(self, dist: Tensor) -> Tensor:
        # dist: (E,)  →  (E, n_gaussians)
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * dist ** 2)


# ─── Continuous Filter Convolution ────────────────────────────────────────────
class CFConv(nn.Module):
    """
    Continuous-filter convolution: the core SchNet operation.

    For each atom i, aggregates information from neighbors j within cutoff:
        h_i' = W_nn(e_ij) ⊙ h_j   (element-wise multiply then sum)

    where e_ij is the Gaussian-expanded pairwise distance and W_nn is an MLP
    that produces the continuous filter.
    """
    def __init__(self, in_channels: int, out_channels: int, n_gaussians: int, n_filters: int):
        super().__init__()
        self.filter_network = nn.Sequential(
            nn.Linear(n_gaussians, n_filters),
            nn.SiLU(),                          # SiLU (Swish) — smooth, non-monotonic
            nn.Linear(n_filters, in_channels),
        )
        self.output_proj = nn.Linear(in_channels, out_channels)

    def forward(self, h: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """
        Args:
            h:          (N, in_channels)  atom features
            edge_index: (2, E)            [source, target] pairs
            edge_attr:  (E, n_gaussians)  Gaussian-expanded distances
        Returns:
            h_new:      (N, out_channels)
        """
        src, tgt = edge_index
        # Compute continuous filter weights from distance basis
        W = self.filter_network(edge_attr)        # (E, in_channels)
        # Gather source atom features and multiply by filter
        msg = h[src] * W                          # (E, in_channels)
        # Aggregate messages at target atoms (sum)
        agg = torch.zeros_like(h).scatter_add_(0, tgt.unsqueeze(-1).expand_as(msg), msg)
        return self.output_proj(agg)              # (N, out_channels)


# ─── SchNet Interaction Block ─────────────────────────────────────────────────
class InteractionBlock(nn.Module):
    """
    Single SchNet interaction block:
        h → CFConv → Dense → Residual

    Each block allows atoms to aggregate information from their neighbors
    within the cutoff radius, weighted by the continuous filter.
    """
    def __init__(self, embedding_dim: int, n_filters: int, n_gaussians: int):
        super().__init__()
        self.cfconv = CFConv(embedding_dim, embedding_dim, n_gaussians, n_filters)
        self.norm1  = nn.LayerNorm(embedding_dim)
        self.dense  = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.norm2  = nn.LayerNorm(embedding_dim)

    def forward(self, h: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # Message passing
        h_conv = self.cfconv(h, edge_index, edge_attr)
        h = self.norm1(h + h_conv)     # Residual connection
        # Dense update
        h_dense = self.dense(h)
        h = self.norm2(h + h_dense)    # Residual connection
        return h


# ─── SchNet Encoder ───────────────────────────────────────────────────────────
class SchNetEncoder(nn.Module):
    """
    Full SchNet encoder: atom embedding + n interaction blocks + global pooling.

    Takes a molecular graph with 3D coordinates and produces a fixed-size
    molecular fingerprint via global sum pooling.
    """
    def __init__(self, config: SchNetConfig):
        super().__init__()
        self.cfg = config

        # Atom embedding: atomic number → dense vector
        self.atom_emb = nn.Embedding(config.max_z, config.embedding_dim, padding_idx=0)

        # Gaussian distance expansion
        self.distance_expansion = GaussianSmearing(
            start=0.0, stop=config.cutoff, n_gaussians=config.n_gaussians
        )

        # Interaction blocks
        self.interactions = nn.ModuleList([
            InteractionBlock(config.embedding_dim, config.n_filters, config.n_gaussians)
            for _ in range(config.n_interactions)
        ])

        # Readout MLP: atom features → per-atom contribution → global sum
        self.readout = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.SiLU(),
            nn.Linear(config.embedding_dim, config.readout_dim),
        )

    def forward(self, data: Data) -> Tensor:
        """
        Args:
            data: PyG Data with:
                .z:          (N,) atomic numbers
                .pos:        (N, 3) 3D coordinates
                .edge_index: (2, E) edges within cutoff
                .batch:      (N,) batch vector
        Returns:
            mol_fp: (B, readout_dim) molecular fingerprints
        """
        # Atom embeddings
        h = self.atom_emb(data.z)           # (N, embedding_dim)

        # Compute pairwise distances for edges
        pos = data.pos
        src, tgt = data.edge_index
        dists = (pos[src] - pos[tgt]).norm(dim=-1)   # (E,)
        edge_attr = self.distance_expansion(dists)    # (E, n_gaussians)

        # Apply interaction blocks
        for interaction in self.interactions:
            h = interaction(h, data.edge_index, edge_attr)

        # Per-atom readout
        atom_contrib = self.readout(h)      # (N, readout_dim)

        # Global sum pooling → molecular fingerprint
        mol_fp = global_add_pool(atom_contrib, data.batch)  # (B, readout_dim)
        return mol_fp


# ─── Multi-Task Prediction Head ───────────────────────────────────────────────
class MultiTaskHead(nn.Module):
    """
    Two parallel prediction heads with shared dropout:
      1. Toxicity Score [0,1] (binary classification via sigmoid)
      2. Hepatic Clearance [mL/min/kg] (regression via ReLU)

    MC Dropout is ACTIVE at both training AND inference time.
    At inference, N=50 stochastic forward passes estimate uncertainty.
    """
    def __init__(self, config: SchNetConfig):
        super().__init__()
        rdim = config.readout_dim
        hdim = config.hidden_dim
        hhdim = config.head_hidden_dim
        p = config.dropout_p

        # Shared dense layer
        self.shared = nn.Sequential(
            nn.Linear(rdim, hdim),
            nn.SiLU(),
            nn.Dropout(p=p),
        )

        # Toxicity branch
        self.tox_head = nn.Sequential(
            nn.Linear(hdim, hhdim),
            nn.SiLU(),
            nn.Dropout(p=p),
            nn.Linear(hhdim, 1),
            nn.Sigmoid(),
        )

        # Hepatic clearance branch (regression, non-negative)
        self.clearance_head = nn.Sequential(
            nn.Linear(hdim, hhdim),
            nn.SiLU(),
            nn.Dropout(p=p),
            nn.Linear(hhdim, 1),
            nn.Softplus(),  # Smooth non-negative activation
        )

    def forward(self, mol_fp: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            tox_score:  (B, 1) ∈ [0, 1]
            clearance:  (B, 1) ∈ [0, ∞)  mL/min/kg
        """
        shared = self.shared(mol_fp)
        tox_score = self.tox_head(shared)
        clearance = self.clearance_head(shared)
        return tox_score, clearance


# ─── Full Sentinel Model ──────────────────────────────────────────────────────
class BiosecuritySentinel(nn.Module):
    """
    Complete Biosecurity Sentinel: SchNet encoder + multi-task head.

    Key behaviors:
    - train(): standard forward pass (single sample per molecule)
    - eval() + mc_predict(): N stochastic passes with Dropout active
    """

    def __init__(self, config: SchNetConfig = None):
        super().__init__()
        self.cfg     = config or SchNetConfig()
        self.encoder = SchNetEncoder(self.cfg)
        self.head    = MultiTaskHead(self.cfg)

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        """Standard forward pass."""
        mol_fp = self.encoder(data)
        return self.head(mol_fp)

    def mc_predict(self, data: Data, n_samples: int = None) -> dict:
        """
        Monte Carlo Dropout inference.

        Runs n_samples stochastic forward passes with dropout ACTIVE,
        returning mean predictions and 95% confidence intervals.

        CRITICAL: Dropout must remain active at inference for MC to work.
        This is achieved by calling model.train() but wrapping in torch.no_grad().
        """
        n = n_samples or self.cfg.n_mc_samples

        # Force dropout ON for MC sampling
        self.train()

        tox_samples  = []
        cler_samples = []

        with torch.no_grad():
            for _ in range(n):
                tox, cler = self.forward(data)
                tox_samples.append(tox.squeeze(-1))   # (B,)
                cler_samples.append(cler.squeeze(-1)) # (B,)

        tox_stack  = torch.stack(tox_samples,  dim=0)   # (n, B)
        cler_stack = torch.stack(cler_samples, dim=0)   # (n, B)

        tox_mean   = tox_stack.mean(dim=0)
        tox_std    = tox_stack.std(dim=0)
        cler_mean  = cler_stack.mean(dim=0)
        cler_std   = cler_stack.std(dim=0)

        # 95% CI (Normal approximation)
        z = 1.96
        tox_lo  = (tox_mean  - z * tox_std).clamp(0, 1)
        tox_hi  = (tox_mean  + z * tox_std).clamp(0, 1)
        cler_lo = (cler_mean - z * cler_std).clamp(min=0)
        cler_hi =  cler_mean + z * cler_std

        return {
            "toxicity_mean":        tox_mean,
            "toxicity_std":         tox_std,
            "toxicity_ci_lo":       tox_lo,
            "toxicity_ci_hi":       tox_hi,
            "clearance_mean":       cler_mean,
            "clearance_std":        cler_std,
            "clearance_ci_lo":      cler_lo,
            "clearance_ci_hi":      cler_hi,
            "n_mc_samples":         n,
        }

    def classify_threat(self, mc_result: dict) -> list[dict]:
        """
        Apply threat classification rules to MC prediction results.

        Classification logic:
          UNKNOWN:    MC std > OOD threshold (model uncertain → don't guess)
          SAFE:       Toxicity < 0.3
          MONITOR:    Toxicity 0.3-0.6 OR high tox but rapid clearance
          HIGH-RISK:  Toxicity >= 0.6 AND low clearance (< 20 mL/min/kg)
        """
        tox   = mc_result["toxicity_mean"].tolist()
        std   = mc_result["toxicity_std"].tolist()
        cler  = mc_result["clearance_mean"].tolist()

        # Normalize to scalar lists
        if isinstance(tox, float):   tox   = [tox]
        if isinstance(std, float):   std   = [std]
        if isinstance(cler, float):  cler  = [cler]

        results = []
        for t, s, c in zip(tox, std, cler):
            # Out-of-distribution check (primary safety gate)
            if s >= self.cfg.ood_std_threshold:
                threat_class  = "UNKNOWN"
                threat_color  = "#9C27B0"
                explanation   = (
                    f"Model uncertainty is high (σ={s:.3f} ≥ {self.cfg.ood_std_threshold}). "
                    "This compound is outside the training distribution. "
                    "Manual expert review is mandatory before any decision."
                )
                action        = "MANDATORY_EXPERT_REVIEW"

            elif t < 0.30:
                threat_class  = "SAFE"
                threat_color  = "#2DC653"
                explanation   = (
                    f"Low predicted toxicity (score={t:.3f}). "
                    "No significant multi-target toxicity signals detected."
                )
                action        = "PROCEED"

            elif t >= 0.60 and c < 20.0:
                threat_class  = "HIGH-RISK"
                threat_color  = "#E63946"
                explanation   = (
                    f"High toxicity (score={t:.3f}) combined with low predicted hepatic "
                    f"clearance ({c:.1f} mL/min/kg). Compound is likely to persist "
                    "systemically and cause toxicity. Immediate review required."
                )
                action        = "ESCALATE_TO_HUMAN_REVIEW"

            elif t >= 0.60 and c >= 50.0:
                threat_class  = "MONITOR"
                threat_color  = "#F4A261"
                explanation   = (
                    f"High toxicity signal (score={t:.3f}) but rapid predicted hepatic "
                    f"clearance ({c:.1f} mL/min/kg) suggests first-pass metabolism may "
                    "neutralize systemic exposure. Requires monitoring and confirmatory assay."
                )
                action        = "FLAG_FOR_REVIEW"

            else:
                threat_class  = "MONITOR"
                threat_color  = "#F4A261"
                explanation   = (
                    f"Moderate toxicity signal (score={t:.3f}). "
                    "Borderline risk — additional in vitro assays recommended."
                )
                action        = "FLAG_FOR_REVIEW"

            results.append({
                "threat_class":       threat_class,
                "threat_color":       threat_color,
                "toxicity_score":     round(t, 4),
                "toxicity_std":       round(s, 4),
                "clearance_mL_min_kg":round(c, 2),
                "is_ood":             s >= self.cfg.ood_std_threshold,
                "explanation":        explanation,
                "required_action":    action,
            })

        return results


# ─── Temperature Scaling (Post-hoc calibration) ───────────────────────────────
class TemperatureScaling(nn.Module):
    """
    Single-parameter calibration wrapper.
    Learns temperature T such that: calibrated_logit = logit / T
    Applied post-training to minimize Expected Calibration Error (ECE).
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature

    def calibrate(self, logits: Tensor, labels: Tensor, n_iter: int = 500) -> float:
        """Fit temperature on validation logits/labels using NLL minimization."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=n_iter)
        nll_loss  = nn.BCEWithLogitsLoss()

        def eval_():
            optimizer.zero_grad()
            loss = nll_loss(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_)
        return self.temperature.item()


# ─── Model Factory ────────────────────────────────────────────────────────────
def build_sentinel(config: SchNetConfig = None) -> BiosecuritySentinel:
    """Build and initialize a Biosecurity Sentinel model."""
    model = BiosecuritySentinel(config or SchNetConfig())
    # Xavier initialization for linear layers
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick architecture sanity check
    from torch_geometric.data import Data

    cfg   = SchNetConfig()
    model = build_sentinel(cfg)
    print(f"Biosecurity Sentinel — {count_parameters(model):,} trainable parameters")

    # Synthetic batch: 2 molecules, 10 atoms each
    N, B = 20, 2
    data = Data(
        z          = torch.randint(1, 9, (N,)),
        pos        = torch.randn(N, 3) * 2.0,
        edge_index = torch.randint(0, N, (2, 40)),
        batch      = torch.tensor([0]*10 + [1]*10),
    )

    # Standard forward pass
    tox, cler = model(data)
    print(f"Forward pass OK | Toxicity: {tox.shape}, Clearance: {cler.shape}")

    # MC Dropout prediction
    mc = model.mc_predict(data, n_samples=10)
    print(f"MC predict OK   | tox_mean={mc['toxicity_mean'].tolist()}, std={mc['toxicity_std'].tolist()}")

    # Threat classification
    classes = model.classify_threat(mc)
    for i, c in enumerate(classes):
        print(f"Molecule {i}: {c['threat_class']} (action: {c['required_action']})")

    print("\nAll checks passed.")
