# Biosecurity Sentinel

Biosecurity Sentinel screens molecules for toxicity, hepatic clearance liability, and dual-use chemical weapons risk. It combines a structural alert pre-screen with a SchNet 3D graph neural network and Monte Carlo Dropout to give you calibrated confidence intervals alongside every prediction — not just a score, but an honest estimate of how certain the model is.

[![Tests](https://github.com/YOUR_USERNAME/biosecurity-sentinel/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR_USERNAME/biosecurity-sentinel/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## How it works

Given a SMILES string, the pipeline runs in four stages:

```
SMILES → Structural Alert Screen → 3D Conformer (ETKDGv3) → SchNet GNN → MC Dropout × 30 → Risk Report
```

The structural alert layer checks first. If a molecule matches a known CWC Schedule 1/2 pharmacophore (organophosphates, nitrogen mustards, etc.) it halts immediately — the GNN never runs. For everything else, SchNet builds a 3D graph from the conformer and runs 30 forward passes with dropout active, giving a mean, standard deviation, and 95% confidence interval for each of three heads: toxicity, hepatic clearance, and dual-use risk.

Risk classification comes out as one of: `CRITICAL`, `HIGH`, `MODERATE`, `LOW`, or `NEGLIGIBLE`. High epistemic uncertainty triggers an `UNKNOWN` flag — the model surfaces what it doesn't know rather than hiding it.

---

## Quickstart

### Docker (easiest)

```bash
git clone https://github.com/YOUR_USERNAME/biosecurity-sentinel
cd biosecurity-sentinel
docker compose -f docker/docker-compose.yml up --build
```

API will be at `http://localhost:8000`.

### Local

```bash
git clone https://github.com/YOUR_USERNAME/biosecurity-sentinel
cd biosecurity-sentinel

# torch-geometric requires matching torch + CUDA versions — read requirements.txt first
pip install torch==2.3.0
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install -r requirements.txt

make serve
```

---

## Usage

```bash
curl -X POST http://localhost:8000/assess \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "name": "Ibuprofen"}'
```

The response includes a risk classification, whether a structural alert fired, and per-head predictions with mean, standard deviation, and 95% CI. See the demo notebook for full output examples.

---

## Demo notebook

Open `notebooks/demo.ipynb` and run all cells. It screens six molecules including caffeine, ibuprofen, and an organophosphate analogue. No trained weights needed — the structural alert layer fires on the organophosphate before the GNN is ever called.

```bash
make demo
```

---

## Training

```bash
# Download Tox21, ToxCast, QM9
make data

# Phase 1: pretrain SchNet encoder on QM9
# Phase 2: fine-tune multi-task heads on Tox21
make train

# Evaluate on held-out test set
make evaluate
```

Results are written to `results/benchmark.json` and `results/benchmark_report.txt`.

---

## Project structure

```
biosecurity-sentinel/
├── .env.template               # Environment variable template — copy to .env
├── .gitignore
├── .github/
│   └── workflows/
│       └── tests.yml           # CI — runs pytest on push
├── LICENSE
├── Makefile                    # install / train / evaluate / test / serve / demo
├── requirements.txt
├── api/
│   └── main.py                 # FastAPI endpoints
├── config/
│   └── settings.py             # Pydantic settings (loads from .env)
├── core/
│   ├── assessment_engine.py    # Main pipeline orchestrator
│   └── molecule_builder.py     # SMILES → 3D graph via ETKDGv3
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/
│   ├── architecture.md
│   └── structural_alerts.md
├── models/
│   ├── schnet_model.py         # SchNet GNN + MC Dropout + multi-task heads
│   └── weights/                # Trained weights go here — not tracked by git
├── notebooks/
│   └── demo.ipynb
├── scripts/
│   └── start.sh
├── tests/
│   └── test_sentinel.py
└── training/
    ├── download_datasets.py    # Fetch Tox21, ToxCast, QM9
    ├── train.py                # Full two-phase training loop
    └── evaluate.py             # Reproduce benchmark numbers
```

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full writeup — SchNet interaction layers, MC Dropout uncertainty quantification, and the 3-fallback conformer generation strategy.

---

## License

MIT — see [LICENSE](LICENSE).
