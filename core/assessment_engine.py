"""
Biosecurity Assessment Engine
Full pipeline: SMILES → structural alerts → 3D conformer → SchNet → risk report
"""

import logging
import json
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, List
from pathlib import Path

import torch
from torch_geometric.data import Batch

from core.molecule_builder import MoleculeGraphBuilder, MoleculeGraphData
from models.schnet_model import BiosecuritySentinelModel, PredictionResult
from config.settings import settings

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Risk classification thresholds
RISK_ORDER = [
    ("CRITICAL", settings.tox_critical_threshold),
    ("HIGH",     settings.tox_high_threshold),
    ("MODERATE", settings.tox_moderate_threshold),
    ("LOW",      settings.tox_low_threshold),
]


def classify_risk(score: float) -> str:
    for label, threshold in RISK_ORDER:
        if score >= threshold:
            return label
    return "NEGLIGIBLE"


class AssessmentEngine:
    """
    High-level assessment interface.
    Handles: structural alert pre-screening, 3D graph construction,
    SchNet inference, uncertainty quantification, risk classification,
    recommendation generation, and audit logging.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.builder = MoleculeGraphBuilder()
        self.model = BiosecuritySentinelModel(
            hidden_channels=settings.schnet_hidden_channels,
            num_interactions=settings.schnet_num_interactions,
            cutoff=settings.schnet_cutoff_angstrom,
            dropout_rate=settings.dropout_rate,
        )

        if model_path and Path(model_path).exists():
            state = torch.load(model_path, map_location=DEVICE)
            self.model.load_state_dict(state)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(
                "No trained model weights found. "
                "Predictions will be random (untrained network). "
                "Train the model first using training/train.py"
            )

        self.model.to(DEVICE)
        self.model.eval()

        # Audit log setup
        settings.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    def assess(
        self,
        smiles: str,
        name: Optional[str] = None,
        requester_id: Optional[str] = None,
        n_mc_samples: int = 30,
    ) -> Dict:
        """
        Full biosecurity assessment of a molecule.

        Pipeline:
          1. Parse SMILES + compute descriptors
          2. Structural alert screening (deterministic — always runs)
          3. 3D conformer generation (3 fallback strategies)
          4. SchNet forward passes × n_mc_samples (MC Dropout)
          5. Risk classification with clearance mitigation check
          6. Recommendation generation
          7. Audit log entry
        """
        ts = datetime.now(timezone.utc).isoformat()
        smiles_hash = hashlib.sha256(smiles.encode()).hexdigest()[:12]

        # ── Step 1–3: Build molecular graph ──────────────────────────────────
        mol_data: MoleculeGraphData = self.builder.build(smiles)

        # ── Step 2 pre-check: CRITICAL structural alert → immediate HALT ─────
        if mol_data.has_critical_alert:
            result = self._build_halt_result(mol_data, name, ts)
            self._audit(result, requester_id, smiles_hash)
            return result

        # ── Conformer failed → return with UNKNOWN classification ────────────
        if not mol_data.conformer_success:
            result = self._build_conformer_failed_result(mol_data, name, ts)
            self._audit(result, requester_id, smiles_hash)
            return result

        # ── Step 4: SchNet inference ──────────────────────────────────────────
        graph = mol_data.graph.to(DEVICE)
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=DEVICE)

        prediction: PredictionResult = self.model.predict_with_uncertainty(
            graph, n_samples=n_mc_samples,
            unknown_threshold=settings.uncertainty_unknown_threshold,
        )

        # ── Step 5: Risk classification ───────────────────────────────────────
        tox_score   = prediction.toxicity_mean
        dual_score  = prediction.dual_use_mean
        clearance   = prediction.clearance_mean
        any_unknown = (prediction.toxicity_unknown or prediction.dual_use_unknown)

        # Rapid hepatic clearance mitigates systemic toxicity risk
        clearance_mitigated = (
            clearance > settings.clearance_mitigation_ml_min_kg
            and tox_score < 0.6
        )

        # Overall risk = max of tox + dual_use signals
        composite_score = max(tox_score, dual_score)

        if any_unknown:
            risk_level = "UNKNOWN"
        elif mol_data.highest_alert_severity in ("HIGH",):
            risk_level = "HIGH"
        elif clearance_mitigated:
            risk_level = f"LOW (rapid clearance: {clearance:.0f} ml/min/kg)"
        else:
            risk_level = classify_risk(composite_score)

        # ── Step 6: Recommendation ────────────────────────────────────────────
        recommendation = self._generate_recommendation(
            risk_level, dual_score, mol_data, clearance_mitigated, any_unknown
        )

        result = {
            "smiles": smiles,
            "canonical_smiles": mol_data.canonical_smiles,
            "name": name,
            "timestamp": ts,
            "status": "ASSESSED",
            "risk_level": risk_level,
            "composite_score": round(composite_score, 4),
            "predictions": prediction.to_dict(),
            "structural_alerts": [
                {
                    "name": a.name,
                    "category": a.category,
                    "severity": a.severity,
                    "smarts": a.smarts,
                    "reference": a.reference,
                }
                for a in mol_data.alert_hits
            ],
            "conformer_fallback_used": mol_data.fallback_used,
            "conformer_method": mol_data.fallback_method,
            "clearance_mitigated": clearance_mitigated,
            "descriptors": mol_data.descriptors,
            "num_heavy_atoms": mol_data.num_heavy_atoms,
            "recommendation": recommendation,
            "mc_samples": n_mc_samples,
        }

        self._audit(result, requester_id, smiles_hash)
        return result

    def assess_batch(self, molecules: List[Dict]) -> Dict:
        """
        Batch assessment of multiple molecules.
        Returns individual results + summary statistics.
        """
        results = []
        for mol in molecules:
            r = self.assess(
                smiles=mol["smiles"],
                name=mol.get("name"),
                requester_id=mol.get("requester_id"),
                n_mc_samples=mol.get("n_mc_samples", settings.n_mc_samples),
            )
            results.append(r)

        risk_counts = {}
        for r in results:
            level = r["risk_level"].split("(")[0].strip()
            risk_counts[level] = risk_counts.get(level, 0) + 1

        return {
            "total": len(results),
            "risk_summary": risk_counts,
            "halt_required": any(r["risk_level"] == "HALT" for r in results),
            "results": results,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_halt_result(self, mol_data: MoleculeGraphData, name, ts) -> Dict:
        critical_alerts = [a for a in mol_data.alert_hits if a.severity == "CRITICAL"]
        return {
            "smiles": mol_data.smiles,
            "canonical_smiles": mol_data.canonical_smiles,
            "name": name,
            "timestamp": ts,
            "status": "HALT",
            "risk_level": "CRITICAL",
            "composite_score": 1.0,
            "predictions": None,
            "structural_alerts": [
                {"name": a.name, "category": a.category, "severity": a.severity,
                 "smarts": a.smarts, "reference": a.reference}
                for a in mol_data.alert_hits
            ],
            "conformer_fallback_used": False,
            "conformer_method": None,
            "clearance_mitigated": False,
            "descriptors": mol_data.descriptors,
            "num_heavy_atoms": mol_data.num_heavy_atoms,
            "recommendation": (
                f"HALT. {len(critical_alerts)} CRITICAL structural alert(s) detected: "
                f"{', '.join(a.name for a in critical_alerts)}. "
                "Do NOT synthesize. Refer to biosafety officer immediately."
            ),
            "mc_samples": 0,
        }

    def _build_conformer_failed_result(self, mol_data: MoleculeGraphData, name, ts) -> Dict:
        return {
            "smiles": mol_data.smiles,
            "canonical_smiles": mol_data.canonical_smiles,
            "name": name,
            "timestamp": ts,
            "status": "CONFORMER_FAILED",
            "risk_level": "UNKNOWN",
            "composite_score": None,
            "predictions": None,
            "structural_alerts": [
                {"name": a.name, "category": a.category, "severity": a.severity,
                 "smarts": a.smarts, "reference": a.reference}
                for a in mol_data.alert_hits
            ],
            "conformer_fallback_used": True,
            "conformer_method": None,
            "clearance_mitigated": False,
            "descriptors": mol_data.descriptors,
            "num_heavy_atoms": mol_data.num_heavy_atoms,
            "recommendation": (
                "3D conformer generation failed for this molecule. "
                "Manual structural review and experimental toxicology required. "
                f"Error: {mol_data.error}"
            ),
            "mc_samples": 0,
        }

    def _generate_recommendation(
        self,
        risk_level: str,
        dual_score: float,
        mol_data: MoleculeGraphData,
        clearance_mitigated: bool,
        any_unknown: bool,
    ) -> str:
        if any_unknown:
            return (
                "Out-of-distribution molecule — model confidence is LOW. "
                "This compound is structurally unlike the training set. "
                "Experimental toxicology testing (in vitro Ames test, MTT assay) "
                "is required before any laboratory use."
            )
        if dual_score >= settings.dual_use_halt_threshold:
            return (
                f"HALT. Dual-use risk score {dual_score:.2f} exceeds threshold {settings.dual_use_halt_threshold}. "
                "Possible weaponisation potential detected. Refer to institutional biosafety committee."
            )
        if mol_data.highest_alert_severity == "HIGH":
            return (
                "HIGH structural alert detected. Exercise extreme caution. "
                "Full safety review required before synthesis. "
                f"Alert: {mol_data.alert_hits[0].name}."
            )
        if "CRITICAL" in risk_level:
            return (
                "High toxicity predicted with high confidence. "
                "Do not synthesize without containment protocols, PPE, and institutional approval."
            )
        if "HIGH" in risk_level:
            return (
                "High toxicity predicted. Detailed experimental ADME/Tox profiling recommended. "
                "Standard lab safety protocols required."
            )
        if "MODERATE" in risk_level:
            return (
                "Moderate toxicity predicted. Standard laboratory precautions apply. "
                "Hepatotoxicity screening (ALT/AST markers) recommended."
            )
        if clearance_mitigated:
            return (
                "Low systemic risk — rapid hepatic clearance predicted. "
                "Monitor for direct tissue toxicity at site of administration."
            )
        return "Low risk profile based on structural and predicted properties. Standard handling protocols apply."

    def _audit(self, result: Dict, requester_id: Optional[str], smiles_hash: str):
        """Append assessment to JSONL audit log for traceability."""
        entry = {
            "timestamp": result.get("timestamp"),
            "smiles_hash": smiles_hash,
            "name": result.get("name"),
            "risk_level": result.get("risk_level"),
            "status": result.get("status"),
            "requester_id": requester_id or "anonymous",
            "alert_count": len(result.get("structural_alerts", [])),
        }
        try:
            with open(settings.audit_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Audit log write failed: {e}")
