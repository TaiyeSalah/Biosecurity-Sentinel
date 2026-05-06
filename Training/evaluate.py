"""
Biosecurity Sentinel — Benchmark Evaluation Script

Reproduces all benchmark numbers reported in the README and docs.
Run after training:
    python training/evaluate.py --model-path models/weights/sentinel.pt

Outputs:
    - results/benchmark.json       Full metric breakdown
    - results/benchmark_report.txt Human-readable summary
    - results/roc_curves/          Per-endpoint ROC curve data (JSON)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.assessment_engine import AssessmentEngine, classify_risk
from core.molecule_builder import MoleculeGraphBuilder
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURAL ALERT BENCHMARK
# Verifies the hard-stop layer fires correctly on known CWC analogues
# and stays silent on benign molecules.
# ═══════════════════════════════════════════════════════════════════════════

STRUCTURAL_ALERT_BENCHMARK = [
    # (smiles, name, expect_halt)
    ("CP(=O)(OCC)F",                    "Organophosphate analogue",     True),
    ("ClCCNCCCl",                        "Nitrogen mustard analogue",     True),
    ("FC(F)(F)C(=O)F",                   "Acyl fluoride (nerve agent-like)", True),
    ("O=C1NC(=O)c2ccccc21",             "Isatoic anhydride",             False),
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   "Caffeine",                     False),
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O",     "Ibuprofen",                    False),
    ("c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34","Pyrene",                       False),
    ("CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34","Testosterone",              False),
]


def evaluate_structural_alerts(engine: AssessmentEngine) -> Dict:
    """
    Evaluate the structural alert pre-screen layer.
    Returns accuracy and per-molecule results.
    """
    logger.info("─" * 60)
    logger.info("STRUCTURAL ALERT LAYER EVALUATION")
    logger.info("─" * 60)

    results = []
    correct = 0

    for smiles, name, expect_halt in STRUCTURAL_ALERT_BENCHMARK:
        result = engine.assess(smiles, name=name)
        triggered = result.get("structural_alert_triggered", False)
        passed = triggered == expect_halt

        status = "✓" if passed else "✗"
        logger.info(
            f"  {status} {name:<40} "
            f"alert={'YES' if triggered else 'NO '} "
            f"(expected={'HALT' if expect_halt else 'PASS'})"
        )

        results.append({
            "name": name,
            "smiles": smiles,
            "expected_halt": expect_halt,
            "triggered": triggered,
            "correct": passed,
        })

        if passed:
            correct += 1

    accuracy = correct / len(STRUCTURAL_ALERT_BENCHMARK)
    logger.info(f"\n  Structural alert accuracy: {accuracy:.1%} ({correct}/{len(STRUCTURAL_ALERT_BENCHMARK)})")

    return {
        "accuracy": accuracy,
        "n_correct": correct,
        "n_total": len(STRUCTURAL_ALERT_BENCHMARK),
        "per_molecule": results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TOX21 TEST SET EVALUATION
# Loads held-out test CSV and computes AUC-ROC for each endpoint
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_tox21_test_set(
    engine: AssessmentEngine,
    test_csv: str,
    output_dir: Path,
) -> Dict:
    """
    Evaluate model on Tox21 held-out test set.
    Requires: data/tox21_test.csv with columns [smiles, tox_label]
    """
    try:
        import pandas as pd
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    except ImportError:
        logger.warning("pandas/sklearn not installed — skipping Tox21 evaluation")
        return {"skipped": True, "reason": "pandas/sklearn not available"}

    if not Path(test_csv).exists():
        logger.warning(f"Test CSV not found: {test_csv} — skipping Tox21 evaluation")
        logger.warning("Run: python training/download_datasets.py first")
        return {"skipped": True, "reason": f"File not found: {test_csv}"}

    logger.info("─" * 60)
    logger.info("TOX21 TEST SET EVALUATION")
    logger.info("─" * 60)

    df = pd.read_csv(test_csv).dropna(subset=["smiles", "tox_label"])
    logger.info(f"  Evaluating on {len(df)} test compounds...")

    y_true, y_pred_mean, y_pred_std = [], [], []
    failed = 0

    for _, row in df.iterrows():
        try:
            result = engine.assess(row["smiles"])
            if "toxicity" in result:
                y_true.append(float(row["tox_label"]))
                y_pred_mean.append(result["toxicity"]["mean"])
                y_pred_std.append(result["toxicity"]["std"])
        except Exception:
            failed += 1

    y_true = np.array(y_true)
    y_pred_mean = np.array(y_pred_mean)
    y_pred_std = np.array(y_pred_std)

    auc = roc_auc_score(y_true, y_pred_mean)
    ap  = average_precision_score(y_true, y_pred_mean)

    threshold = 0.5
    y_binary = (y_pred_mean >= threshold).astype(int)
    f1 = f1_score(y_true, y_binary, zero_division=0)

    # Calibration: how often does high uncertainty correlate with errors?
    errors = np.abs(y_true - y_pred_mean)
    uncertainty_correlation = float(np.corrcoef(y_pred_std, errors)[0, 1])

    # Save ROC curve data
    roc_dir = output_dir / "roc_curves"
    roc_dir.mkdir(parents=True, exist_ok=True)

    try:
        from sklearn.metrics import roc_curve, precision_recall_curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_mean)
        roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc)}
        with open(roc_dir / "tox21_roc.json", "w") as f:
            json.dump(roc_data, f)
    except Exception:
        pass

    logger.info(f"  AUC-ROC:               {auc:.4f}  (target ≥ 0.82)")
    logger.info(f"  Average Precision:     {ap:.4f}")
    logger.info(f"  F1 (threshold=0.5):    {f1:.4f}")
    logger.info(f"  Uncertainty-Error Corr:{uncertainty_correlation:.4f}  (positive = calibrated)")
    logger.info(f"  Failed molecules:      {failed}")

    return {
        "n_evaluated": len(y_true),
        "n_failed": failed,
        "auc_roc": float(auc),
        "average_precision": float(ap),
        "f1_score": float(f1),
        "uncertainty_error_correlation": uncertainty_correlation,
        "target_auc": 0.82,
        "meets_target": auc >= 0.82,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MC DROPOUT CALIBRATION
# Verifies that uncertainty estimates are meaningful, not noise
# ═══════════════════════════════════════════════════════════════════════════

# Molecules where the model SHOULD be uncertain (edge cases)
UNCERTAIN_MOLECULES = [
    ("C1CCCCC1",        "Cyclohexane",     "low_tox"),    # should be low, confident
    ("c1ccccc1",        "Benzene",         "ambiguous"),   # carcinogen but mechanism varies
    ("ClC(Cl)(Cl)Cl",  "Carbon tetrachloride", "high_tox"),
]

def evaluate_mc_dropout_calibration(engine: AssessmentEngine) -> Dict:
    """
    Check that MC Dropout uncertainty is meaningful:
    - Ambiguous molecules should have higher std than clear-cut ones
    - Confidence intervals should be appropriately wide vs narrow
    """
    logger.info("─" * 60)
    logger.info("MC DROPOUT CALIBRATION CHECK")
    logger.info("─" * 60)

    results = []
    for smiles, name, expected_certainty in UNCERTAIN_MOLECULES:
        result = engine.assess(smiles, name=name, n_mc_samples=50)
        if "toxicity" not in result:
            continue
        tox = result["toxicity"]
        ci_width = tox.get("ci_95_high", 0) - tox.get("ci_95_low", 0)
        logger.info(
            f"  {name:<30} mean={tox['mean']:.3f} "
            f"std={tox['std']:.3f} CI_width={ci_width:.3f} "
            f"[expected: {expected_certainty}]"
        )
        results.append({
            "name": name,
            "smiles": smiles,
            "expected_certainty": expected_certainty,
            "tox_mean": tox["mean"],
            "tox_std": tox["std"],
            "ci_width": ci_width,
        })

    return {"molecules": results}


# ═══════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(results: Dict, output_path: Path) -> str:
    """Generate human-readable benchmark report."""
    lines = [
        "=" * 70,
        "BIOSECURITY SENTINEL — BENCHMARK EVALUATION REPORT",
        "=" * 70,
        "",
    ]

    # Structural alerts
    sa = results.get("structural_alerts", {})
    if sa:
        lines += [
            "1. STRUCTURAL ALERT LAYER",
            f"   Accuracy: {sa.get('accuracy', 0):.1%} ({sa.get('n_correct', 0)}/{sa.get('n_total', 0)})",
            f"   Target:   100%",
            f"   Status:   {'✓ PASS' if sa.get('accuracy', 0) >= 1.0 else '✗ FAIL'}",
            "",
        ]

    # Tox21
    t21 = results.get("tox21", {})
    if t21 and not t21.get("skipped"):
        lines += [
            "2. TOX21 TEST SET",
            f"   AUC-ROC:   {t21.get('auc_roc', 0):.4f}",
            f"   Target:    ≥ 0.82",
            f"   Status:    {'✓ PASS' if t21.get('meets_target') else '✗ FAIL (more training needed)'}",
            f"   Avg Prec:  {t21.get('average_precision', 0):.4f}",
            f"   F1:        {t21.get('f1_score', 0):.4f}",
            f"   UQ Corr:   {t21.get('uncertainty_error_correlation', 0):.4f}  (>0 = calibrated UQ)",
            "",
        ]
    elif t21.get("skipped"):
        lines += [
            "2. TOX21 TEST SET",
            f"   SKIPPED — {t21.get('reason', 'unknown')}",
            "",
        ]

    # MC Dropout
    mc = results.get("mc_dropout", {})
    if mc:
        lines += ["3. MC DROPOUT CALIBRATION", "   (Visual inspection — see results/benchmark.json for values)"]
        for mol in mc.get("molecules", []):
            lines.append(
                f"   {mol['name']:<30} mean={mol['tox_mean']:.3f} "
                f"std={mol['tox_std']:.3f} [{mol['expected_certainty']}]"
            )
        lines.append("")

    lines += ["=" * 70, "END OF REPORT", "=" * 70]
    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    return report


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Biosecurity Sentinel benchmarks")
    parser.add_argument("--model-path", default="models/weights/sentinel.pt",
                        help="Path to trained model weights (.pt)")
    parser.add_argument("--tox21-test-csv", default="data/tox21_test.csv",
                        help="Path to held-out Tox21 test CSV")
    parser.add_argument("--output", default="results/benchmark.json",
                        help="Output path for full JSON results")
    parser.add_argument("--skip-tox21", action="store_true",
                        help="Skip Tox21 test set eval (use if no test CSV available)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BIOSECURITY SENTINEL — BENCHMARK EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Model:  {args.model_path}")
    logger.info(f"Device: {DEVICE}")
    logger.info("")

    engine = AssessmentEngine(model_path=args.model_path)

    all_results = {}

    # 1. Structural alerts
    all_results["structural_alerts"] = evaluate_structural_alerts(engine)

    # 2. Tox21 test set
    if not args.skip_tox21:
        all_results["tox21"] = evaluate_tox21_test_set(engine, args.tox21_test_csv, output_dir)
    else:
        all_results["tox21"] = {"skipped": True, "reason": "--skip-tox21 flag set"}

    # 3. MC Dropout calibration
    all_results["mc_dropout"] = evaluate_mc_dropout_calibration(engine)

    # Save full JSON results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n✓ Full results saved to {output_path}")

    # Generate human-readable report
    report_path = output_dir / "benchmark_report.txt"
    report = generate_report(all_results, report_path)
    logger.info(f"✓ Report saved to {report_path}")
    logger.info("")
    print(report)
