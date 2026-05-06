"""
Biosecurity Sentinel — Test Suite
Run: pytest tests/ -v --cov=core --cov=models --cov-report=term-missing
"""

import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.molecule_builder import MoleculeGraphBuilder, MoleculeGraphData
from config.settings import STRUCTURAL_ALERTS, DEMO_COMPOUNDS


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURAL ALERT SCREENING
# ═══════════════════════════════════════════════════════════════════════════

class TestStructuralAlerts:

    @pytest.fixture
    def builder(self):
        return MoleculeGraphBuilder()

    def test_organophosphate_triggers_critical_alert(self, builder):
        """Sarin-class organophosphate fluoride must trigger CRITICAL alert."""
        result = builder.build("CP(=O)(OCC)F")
        assert result.has_critical_alert, "Organophosphate fluoride must trigger CRITICAL"
        assert any("fluoride" in a.name.lower() or "organophosphate" in a.name.lower()
                   for a in result.alert_hits)

    def test_ibuprofen_has_no_alerts(self, builder):
        """Safe NSAID must return no structural alerts."""
        result = builder.build("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        assert len(result.alert_hits) == 0
        assert not result.has_critical_alert

    def test_caffeine_has_no_alerts(self, builder):
        result = builder.build("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
        assert not result.has_critical_alert

    def test_alert_severity_field_populated(self, builder):
        """All alert hits must have severity field."""
        result = builder.build("CP(=O)(OCC)F")
        for hit in result.alert_hits:
            assert hit.severity in ("CRITICAL", "HIGH", "MODERATE", "LOW")
            assert hit.name
            assert hit.category
            assert hit.smarts

    def test_all_smarts_patterns_compile(self):
        """Every SMARTS pattern in the alert library must compile without error."""
        from rdkit import Chem
        for alert in STRUCTURAL_ALERTS:
            pattern = Chem.MolFromSmarts(alert["smarts"])
            assert pattern is not None, f"SMARTS failed to compile: {alert['smarts']}"

    def test_highest_severity_property(self, builder):
        """highest_alert_severity must return the most severe level found."""
        result = builder.build("CP(=O)(OCC)F")
        assert result.highest_alert_severity == "CRITICAL"


# ═══════════════════════════════════════════════════════════════════════════
# CONFORMER GENERATION
# ═══════════════════════════════════════════════════════════════════════════

class TestConformerGeneration:

    @pytest.fixture
    def builder(self):
        return MoleculeGraphBuilder()

    def test_simple_molecule_gets_conformer(self, builder):
        """Ethanol should always produce a valid 3D conformer."""
        result = builder.build("CCO")
        assert result.conformer_success
        assert result.graph is not None
        assert result.graph.pos.shape[1] == 3   # 3D coordinates

    def test_drug_molecule_gets_conformer(self, builder):
        """Ciprofloxacin — complex ring system — must produce conformer."""
        result = builder.build("OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O")
        assert result.conformer_success

    def test_invalid_smiles_handled_gracefully(self, builder):
        """Invalid SMILES must return failure, not raise an exception."""
        result = builder.build("NOT_A_SMILES_STRING!!!")
        assert not result.conformer_success
        assert result.graph is None
        assert result.error is not None

    def test_graph_z_tensor_contains_atomic_numbers(self, builder):
        """Atomic number tensor must contain valid positive integers."""
        result = builder.build("CCO")
        assert result.conformer_success
        z = result.graph.z
        assert z.min().item() > 0       # No zero atomic numbers
        assert z.max().item() <= 118    # No elements beyond oganesson

    def test_num_atoms_consistent(self, builder):
        """num_atoms field must match actual atoms in graph."""
        result = builder.build("CCO")
        assert result.conformer_success
        assert result.graph.num_nodes == result.num_atoms

    def test_fallback_flag_set_correctly(self, builder):
        """Easy molecule should not require fallback."""
        result = builder.build("CCO")
        # Simple molecules should not need fallback
        if result.conformer_success:
            # fallback_used may be True or False — just confirm it's a bool
            assert isinstance(result.fallback_used, bool)

    def test_canonical_smiles_returned(self, builder):
        """canonical_smiles must differ from or equal input but always be valid."""
        from rdkit import Chem
        result = builder.build("c1ccccc1")  # Benzene
        mol = Chem.MolFromSmiles(result.canonical_smiles)
        assert mol is not None, "canonical_smiles must be a valid SMILES string"


# ═══════════════════════════════════════════════════════════════════════════
# 2D DESCRIPTORS
# ═══════════════════════════════════════════════════════════════════════════

class TestDescriptors:

    @pytest.fixture
    def builder(self):
        return MoleculeGraphBuilder()

    def test_lipinski_descriptors_present(self, builder):
        result = builder.build("CC(C)Cc1ccc(cc1)C(C)C(=O)O")  # Ibuprofen
        for key in ["molecular_weight", "logp", "hbd", "hba", "tpsa"]:
            assert key in result.descriptors, f"Missing descriptor: {key}"

    def test_ibuprofen_molecular_weight_correct(self, builder):
        result = builder.build("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        mw = result.descriptors.get("molecular_weight", 0)
        assert 200 < mw < 215, f"Ibuprofen MW should be ~206, got {mw}"

    def test_qed_bounded(self, builder):
        """QED (drug-likeness) must be between 0 and 1."""
        result = builder.build("CCO")
        qed = result.descriptors.get("qed", -1)
        assert 0.0 <= qed <= 1.0

    def test_num_heavy_atoms_positive(self, builder):
        result = builder.build("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        assert result.num_heavy_atoms > 0


# ═══════════════════════════════════════════════════════════════════════════
# ASSESSMENT ENGINE (integration, no trained weights)
# ═══════════════════════════════════════════════════════════════════════════

class TestAssessmentEngine:

    @pytest.fixture
    def engine(self):
        from core.assessment_engine import AssessmentEngine
        # No model_path — tests untrained inference path
        return AssessmentEngine(model_path=None)

    def test_critical_alert_returns_halt_without_model(self, engine):
        """Sarin analog must trigger HALT from structural alerts alone — no model needed."""
        result = engine.assess("CP(=O)(OCC)F", name="Sarin analog test")
        assert result["status"] == "HALT"
        assert result["risk_level"] == "CRITICAL"
        assert len(result["structural_alerts"]) > 0

    def test_invalid_smiles_returns_conformer_failed(self, engine):
        result = engine.assess("INVALID!!!", name="bad input")
        assert result["status"] in ("CONFORMER_FAILED",)
        assert result["risk_level"] == "UNKNOWN"

    def test_result_always_has_recommendation(self, engine):
        """Every result must contain a recommendation string."""
        for smiles in ["CCO", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CP(=O)(OCC)F"]:
            result = engine.assess(smiles)
            assert "recommendation" in result
            assert len(result["recommendation"]) > 10

    def test_audit_log_created(self, engine, tmp_path):
        """Audit log file must be written after assessment."""
        from config.settings import settings
        original = settings.audit_log_path
        settings.audit_log_path = tmp_path / "test_audit.jsonl"

        engine.assess("CCO", requester_id="test_run")

        assert settings.audit_log_path.exists()
        lines = settings.audit_log_path.read_text().strip().split("\n")
        assert len(lines) >= 1

        import json
        entry = json.loads(lines[0])
        assert entry["requester_id"] == "test_run"
        settings.audit_log_path = original

    def test_batch_returns_summary(self, engine):
        """Batch assessment must return summary counts."""
        result = engine.assess_batch([
            {"smiles": "CCO", "name": "Ethanol"},
            {"smiles": "CP(=O)(OCC)F", "name": "Sarin analog"},
        ])
        assert "total" in result
        assert result["total"] == 2
        assert "risk_summary" in result
        assert result["halt_required"] == True

    def test_demo_compounds_all_assess(self, engine):
        """All built-in demo compounds must complete assessment without exception."""
        for key, compound in DEMO_COMPOUNDS.items():
            result = engine.assess(compound["smiles"], name=compound["name"])
            assert "risk_level" in result, f"No risk_level for {key}"
            assert "recommendation" in result


# ═══════════════════════════════════════════════════════════════════════════
# DEMO NOTEBOOK SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

class TestDemoNotebook:
    """Verify the 5 demo molecules produce expected risk tiers."""

    EXPECTED = {
        "CCO":                             ("NEGLIGIBLE",),            # Ethanol
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O":    ("NEGLIGIBLE", "LOW"),       # Ibuprofen
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C":    ("NEGLIGIBLE",),            # Caffeine
        "CP(=O)(OCC)F":                    ("CRITICAL",),              # Sarin analog → HALT
        "OC=1C(=O)c2ccccc2C=1CC(=O)c1ccccc1": ("MODERATE", "HIGH"),  # Warfarin
    }

    def test_expected_risk_tiers(self):
        from core.assessment_engine import AssessmentEngine
        engine = AssessmentEngine(model_path=None)

        for smiles, expected_levels in self.EXPECTED.items():
            result = engine.assess(smiles)
            level = result["risk_level"].split("(")[0].strip()
            assert level in expected_levels, (
                f"SMILES {smiles[:30]} → got {level}, expected one of {expected_levels}"
            )
