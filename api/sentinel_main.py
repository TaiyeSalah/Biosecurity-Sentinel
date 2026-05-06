"""
Biosecurity Sentinel & Threat Forecaster
FastAPI Inference API — main.py

Endpoints:
  POST /predict          — single SMILES prediction
  POST /predict/batch    — batch SMILES prediction
  GET  /review-queue     — list pending HIGH-RISK reviews
  POST /review/{id}      — submit human review decision
  GET  /model/info       — model version and calibration stats
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from model import BiosecuritySentinel, SchNetConfig, build_sentinel
from data_pipeline import smiles_to_graph, smiles_list_to_batch

log = logging.getLogger("sentinel.api")
logging.basicConfig(level=logging.INFO)

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH   = os.getenv("SENTINEL_MODEL_PATH", "checkpoints/sentinel_finetuned.pt")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REVIEW_STORE = []   # In-memory for dev; replace with DB in production

# ─── Global model ─────────────────────────────────────────────────────────────
sentinel: BiosecuritySentinel = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sentinel
    sentinel = build_sentinel(SchNetConfig())
    if Path(MODEL_PATH).exists():
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        sentinel.load_state_dict(ckpt["model_state_dict"])
        log.info(f"Loaded model from {MODEL_PATH}")
    else:
        log.warning(f"No checkpoint at {MODEL_PATH} — using random weights (dev mode)")
    sentinel = sentinel.to(DEVICE)
    log.info(f"Biosecurity Sentinel ready on {DEVICE}")
    yield

app = FastAPI(
    title="Biosecurity Sentinel & Threat Forecaster",
    version="1.0.0",
    description="3D-aware molecular threat classification with uncertainty quantification",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Pydantic models ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the molecule to screen")
    n_mc_samples: int = Field(50, ge=10, le=200, description="Number of MC Dropout passes")
    submitter_id: Optional[str] = Field(None, description="For audit trail")

class PredictBatchRequest(BaseModel):
    smiles_list: List[str] = Field(..., max_items=100)
    n_mc_samples: int = Field(50, ge=10, le=200)
    submitter_id: Optional[str] = None

class ReviewDecision(BaseModel):
    decision: str = Field(..., description="CLEARED | ESCALATED | FALSE_POSITIVE")
    reviewer_id: str
    notes: Optional[str] = None

class ThreatResult(BaseModel):
    query_id: str
    smiles: str
    threat_class: str
    threat_color: str
    toxicity_score: float
    toxicity_std: float
    toxicity_ci_lo: float
    toxicity_ci_hi: float
    clearance_mL_min_kg: float
    clearance_ci_lo: float
    clearance_ci_hi: float
    is_ood: bool
    explanation: str
    required_action: str
    n_mc_samples: int
    inference_ms: float
    model_version: str
    timestamp: str

# ─── Core prediction logic ────────────────────────────────────────────────────
def _run_prediction(smiles: str, n_mc: int, submitter_id: str = None) -> dict:
    t0 = time.perf_counter()

    graph = smiles_to_graph(smiles, use_pubchem_fallback=True)
    if graph is None:
        raise HTTPException(
            422,
            detail=f"Could not generate 3D conformer for SMILES: '{smiles[:80]}'. "
                   "Check that the SMILES is valid. If the compound is highly exotic, "
                   "try providing coordinates directly via the /predict/3d endpoint."
        )

    graph = graph.to(DEVICE)

    # Add batch dimension (single molecule)
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([graph])

    mc_result     = sentinel.mc_predict(batch, n_samples=n_mc)
    classifications = sentinel.classify_threat(mc_result)
    cls           = classifications[0]

    elapsed_ms = (time.perf_counter() - t0) * 1000

    query_id = str(uuid.uuid4())[:8]
    result = {
        "query_id":           query_id,
        "smiles":             smiles,
        "threat_class":       cls["threat_class"],
        "threat_color":       cls["threat_color"],
        "toxicity_score":     cls["toxicity_score"],
        "toxicity_std":       cls["toxicity_std"],
        "toxicity_ci_lo":     float(mc_result["toxicity_ci_lo"].item()),
        "toxicity_ci_hi":     float(mc_result["toxicity_ci_hi"].item()),
        "clearance_mL_min_kg":cls["clearance_mL_min_kg"],
        "clearance_ci_lo":    float(mc_result["clearance_ci_lo"].item()),
        "clearance_ci_hi":    float(mc_result["clearance_ci_hi"].item()),
        "is_ood":             cls["is_ood"],
        "explanation":        cls["explanation"],
        "required_action":    cls["required_action"],
        "n_mc_samples":       n_mc,
        "inference_ms":       round(elapsed_ms, 1),
        "model_version":      "sentinel-v1.0",
        "timestamp":          datetime.utcnow().isoformat(),
        "submitter_id":       submitter_id,
    }

    # Auto-queue HIGH-RISK predictions for human review
    if cls["threat_class"] in ("HIGH-RISK", "UNKNOWN"):
        REVIEW_STORE.append({**result, "review_status": "PENDING"})
        log.warning(f"[REVIEW QUEUE] {cls['threat_class']} prediction logged: {smiles[:50]}")

    return result

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=ThreatResult)
async def predict_single(req: PredictRequest):
    """Screen a single molecule for biosecurity threat."""
    return _run_prediction(req.smiles, req.n_mc_samples, req.submitter_id)

@app.post("/predict/batch")
async def predict_batch(req: PredictBatchRequest):
    """Screen a batch of molecules (max 100)."""
    results = []
    errors  = []
    for smi in req.smiles_list:
        try:
            results.append(_run_prediction(smi, req.n_mc_samples, req.submitter_id))
        except HTTPException as e:
            errors.append({"smiles": smi, "error": e.detail})
    return {
        "results": results,
        "errors":  errors,
        "n_success": len(results),
        "n_failed":  len(errors),
    }

@app.get("/review-queue")
async def get_review_queue(status: str = "PENDING"):
    """Return all predictions pending human review."""
    return {
        "items": [r for r in REVIEW_STORE if r.get("review_status") == status],
        "total": len(REVIEW_STORE),
    }

@app.post("/review/{query_id}")
async def submit_review(query_id: str, decision: ReviewDecision):
    """Submit a human review decision for a queued prediction."""
    for item in REVIEW_STORE:
        if item["query_id"] == query_id:
            item["review_status"]  = "REVIEWED"
            item["review_decision"]= decision.decision
            item["reviewer_id"]    = decision.reviewer_id
            item["review_notes"]   = decision.notes
            item["reviewed_at"]    = datetime.utcnow().isoformat()
            log.info(f"Review submitted: {query_id} → {decision.decision} by {decision.reviewer_id}")
            return {"status": "ok", "query_id": query_id, "decision": decision.decision}
    raise HTTPException(404, f"Query ID {query_id} not found in review queue")

@app.get("/model/info")
async def model_info():
    n_params = sum(p.numel() for p in sentinel.parameters() if p.requires_grad)
    return {
        "model":             "BiosecuritySentinel",
        "version":           "sentinel-v1.0",
        "architecture":      "SchNet (3D GNN) + MC Dropout Multi-Task Head",
        "n_parameters":      f"{n_params:,}",
        "device":            str(DEVICE),
        "n_interaction_blocks": sentinel.cfg.n_interactions,
        "n_filters":         sentinel.cfg.n_filters,
        "cutoff_angstrom":   sentinel.cfg.cutoff,
        "mc_dropout_p":      sentinel.cfg.dropout_p,
        "ood_threshold":     sentinel.cfg.ood_std_threshold,
        "training_datasets": ["Tox21 (NIH)", "ClinTox", "QM9 (pre-training)"],
        "model_path":        MODEL_PATH,
        "checkpoint_exists": Path(MODEL_PATH).exists(),
    }

@app.get("/health")
async def health():
    return {"status": "ok", "service": "biosecurity-sentinel", "timestamp": datetime.utcnow().isoformat()}
