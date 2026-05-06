"""
Molecular Graph Builder
SMILES → 3D Conformer → PyTorch Geometric Data object

Fallback hierarchy (never silently skips a molecule):
  1. ETKDGv3 + MMFF94 minimisation
  2. ETKDGv3 with random seed variation
  3. Classic distance geometry (ETDG)
  4. Flag as CONFORMER_FAILED — returned explicitly, never swallowed
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import rdDistGeom

from config.settings import STRUCTURAL_ALERTS

logger = logging.getLogger(__name__)


@dataclass
class AlertHit:
    smarts: str
    name: str
    category: str
    severity: str
    reference: str


@dataclass
class MoleculeGraphData:
    smiles: str
    canonical_smiles: str
    graph: Optional[Data]
    conformer_success: bool
    fallback_used: bool
    fallback_method: Optional[str]
    alert_hits: List[AlertHit]
    descriptors: Dict[str, float]
    num_atoms: int
    num_heavy_atoms: int
    error: Optional[str] = None

    @property
    def has_critical_alert(self) -> bool:
        return any(a.severity == "CRITICAL" for a in self.alert_hits)

    @property
    def highest_alert_severity(self) -> Optional[str]:
        order = ["CRITICAL", "HIGH", "MODERATE", "LOW"]
        for sev in order:
            if any(a.severity == sev for a in self.alert_hits):
                return sev
        return None


# Precompile SMARTS patterns once at module load
_COMPILED_ALERTS = []
for alert in STRUCTURAL_ALERTS:
    pattern = Chem.MolFromSmarts(alert["smarts"])
    if pattern is not None:
        _COMPILED_ALERTS.append((pattern, alert))
    else:
        logger.warning(f"Failed to compile SMARTS: {alert['smarts']}")


class MoleculeGraphBuilder:
    """
    Converts SMILES strings to 3D molecular graphs for SchNet.

    SchNet requires:
      - z: atomic numbers (long tensor)
      - pos: 3D Cartesian coordinates in Ångströms (float tensor)

    Hydrogen atoms are included for accurate 3D geometry.
    """

    def build(self, smiles: str) -> MoleculeGraphData:
        """Main entry point. Always returns a MoleculeGraphData — never raises."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return MoleculeGraphData(
                smiles=smiles, canonical_smiles=smiles,
                graph=None, conformer_success=False,
                fallback_used=False, fallback_method=None,
                alert_hits=[], descriptors={},
                num_atoms=0, num_heavy_atoms=0,
                error=f"RDKit failed to parse SMILES: {smiles!r}"
            )

        canonical = Chem.MolToSmiles(mol)
        mol_h = Chem.AddHs(mol)

        alerts = self._screen_alerts(mol)
        descriptors = self._compute_descriptors(mol)
        num_heavy = mol.GetNumAtoms()
        num_atoms = mol_h.GetNumAtoms()

        graph, success, fallback_used, fallback_method, error = \
            self._generate_conformer(mol_h)

        return MoleculeGraphData(
            smiles=smiles,
            canonical_smiles=canonical,
            graph=graph,
            conformer_success=success,
            fallback_used=fallback_used,
            fallback_method=fallback_method,
            alert_hits=alerts,
            descriptors=descriptors,
            num_atoms=num_atoms,
            num_heavy_atoms=num_heavy,
            error=error,
        )

    def _generate_conformer(
        self, mol_h
    ) -> Tuple[Optional[Data], bool, bool, Optional[str], Optional[str]]:
        """
        Try conformer generation strategies in order.
        Returns: (graph, success, fallback_used, method_name, error_msg)
        """

        # Strategy 1: ETKDGv3 + MMFF94 (best quality)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        params.enforceChirality = True
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True

        result = AllChem.EmbedMolecule(mol_h, params)
        if result == 0:
            ff_result = AllChem.MMFFOptimizeMolecule(mol_h, maxIters=2000)
            if ff_result in (0, 1):  # 0=converged, 1=not converged but OK
                return self._mol_to_graph(mol_h), True, False, "ETKDGv3+MMFF94", None

        # Strategy 2: ETKDGv3 with random seed
        logger.debug("ETKDGv3 seed=42 failed, trying random seed")
        params2 = rdDistGeom.ETKDGv3()
        params2.randomSeed = 0
        result2 = AllChem.EmbedMolecule(mol_h, params2)
        if result2 == 0:
            AllChem.UFFOptimizeMolecule(mol_h)
            return self._mol_to_graph(mol_h), True, True, "ETKDGv3_random_seed", None

        # Strategy 3: Classic ETDG distance geometry
        logger.debug("ETKDGv3 failed, trying classic ETDG")
        params3 = rdDistGeom.ETDG()
        result3 = AllChem.EmbedMolecule(mol_h, params3)
        if result3 == 0:
            logger.warning("Used ETDG fallback — conformer quality may be lower")
            return self._mol_to_graph(mol_h), True, True, "ETDG_fallback", None

        # All strategies failed
        err = "All 3 conformer generation strategies failed. Molecule may be highly strained or exotic."
        logger.error(err)
        return None, False, False, None, err

    def _mol_to_graph(self, mol_h) -> Data:
        """Convert RDKit mol (with 3D conformer) to PyG Data object."""
        conf = mol_h.GetConformer()

        z = torch.tensor(
            [atom.GetAtomicNum() for atom in mol_h.GetAtoms()],
            dtype=torch.long,
        )

        pos = torch.tensor(
            [
                [conf.GetAtomPosition(i).x,
                 conf.GetAtomPosition(i).y,
                 conf.GetAtomPosition(i).z]
                for i in range(mol_h.GetNumAtoms())
            ],
            dtype=torch.float,
        )

        return Data(z=z, pos=pos, num_nodes=mol_h.GetNumAtoms())

    def _screen_alerts(self, mol) -> List[AlertHit]:
        """Deterministic structural alert screening against SMARTS library."""
        hits = []
        for pattern, alert_dict in _COMPILED_ALERTS:
            if mol.HasSubstructMatch(pattern):
                hits.append(AlertHit(
                    smarts=alert_dict["smarts"],
                    name=alert_dict["name"],
                    category=alert_dict["category"],
                    severity=alert_dict["severity"],
                    reference=alert_dict["reference"],
                ))
        return hits

    def _compute_descriptors(self, mol) -> Dict[str, float]:
        """Compute Lipinski + ADME-relevant 2D descriptors."""
        try:
            return {
                "molecular_weight":    round(Descriptors.MolWt(mol), 2),
                "logp":                round(Descriptors.MolLogP(mol), 3),
                "hbd":                 rdMolDescriptors.CalcNumHBD(mol),
                "hba":                 rdMolDescriptors.CalcNumHBA(mol),
                "tpsa":                round(rdMolDescriptors.CalcTPSA(mol), 2),
                "rotatable_bonds":     rdMolDescriptors.CalcNumRotatableBonds(mol),
                "num_rings":           rdMolDescriptors.CalcNumRings(mol),
                "num_aromatic_rings":  rdMolDescriptors.CalcNumAromaticRings(mol),
                "num_stereo_centers":  len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
                "qed":                 round(Descriptors.qed(mol), 4),
                "fsp3":                round(rdMolDescriptors.CalcFractionCSP3(mol), 4),
            }
        except Exception as e:
            logger.warning(f"Descriptor computation partial failure: {e}")
            return {}
