"""
Dataset Download Helper
Downloads and prepares Tox21, ToxCast, and QM9 for training.

Usage:
    python training/download_datasets.py --dataset tox21
    python training/download_datasets.py --dataset qm9
    python training/download_datasets.py --all
"""

import argparse
import logging
import urllib.request
import zipfile
import gzip
import shutil
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("./data")


def download_tox21(force: bool = False):
    """
    Download Tox21 challenge dataset from NCI.
    12,000 compounds × 12 toxicity endpoints.
    Saves to data/tox21_processed.csv
    """
    out_path = DATA_DIR / "tox21_processed.csv"
    if out_path.exists() and not force:
        logger.info(f"Tox21 already exists at {out_path}. Use --force to redownload.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Tox21 is available via DeepChem (easiest path)
    try:
        import deepchem as dc
        logger.info("Downloading Tox21 via DeepChem...")
        tasks, datasets, _ = dc.molnet.load_tox21(featurizer="Raw", splitter=None)
        train, _, _ = datasets

        rows = []
        for X, y, w, ids in train.itersamples():
            row = {"smiles": ids, "tox_label": int(y[0] > 0) if y[0] is not None else 0}
            for i, task in enumerate(tasks):
                row[f"task_{task}"] = float(y[i]) if y[i] is not None else None
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        logger.info(f"Tox21 saved: {len(df)} compounds → {out_path}")

    except ImportError:
        logger.warning(
            "DeepChem not installed. Install with: pip install deepchem\n"
            "Alternatively, download manually from:\n"
            "https://tripod.nih.gov/tox21/challenge/download.jsp"
        )


def download_qm9(force: bool = False):
    """
    Download QM9 dataset via PyTorch Geometric.
    130,831 small organic molecules with quantum chemical properties.
    Used for SchNet encoder pretraining.
    """
    out_dir = DATA_DIR / "qm9"
    if out_dir.exists() and not force:
        logger.info(f"QM9 already exists at {out_dir}. Use --force to redownload.")
        return

    try:
        from torch_geometric.datasets import QM9
        logger.info("Downloading QM9 via PyTorch Geometric (~1.7GB)...")
        dataset = QM9(root=str(out_dir))
        logger.info(f"QM9 downloaded: {len(dataset)} molecules → {out_dir}")
    except ImportError:
        logger.error(
            "torch_geometric not installed. Install with:\n"
            "pip install torch-geometric"
        )


def download_zinc20_subset(n_molecules: int = 50000, force: bool = False):
    """
    Download a subset of ZINC20 as negative examples (drug-like, presumed non-toxic).
    Uses the ZINC20 tranche download system.
    """
    out_path = DATA_DIR / "zinc20_subset.csv"
    if out_path.exists() and not force:
        logger.info(f"ZINC20 subset already exists at {out_path}")
        return

    logger.info(
        f"ZINC20 download requires manual tranche selection.\n"
        f"Visit: https://zinc20.docking.org/tranches/home/\n"
        f"Download a 'Drug-like' tranche as SMILES and save to: {out_path}\n"
        f"Column headers should include: smiles"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download training datasets")
    parser.add_argument("--dataset", choices=["tox21", "qm9", "zinc20"],
                        help="Specific dataset to download")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")
    args = parser.parse_args()

    if args.all or args.dataset == "tox21":
        download_tox21(force=args.force)

    if args.all or args.dataset == "qm9":
        download_qm9(force=args.force)

    if args.all or args.dataset == "zinc20":
        download_zinc20_subset(force=args.force)
