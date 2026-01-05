"""
Loader minimale per inference (modello + scaler).

Obiettivo:
- semplice e “da produzione”
- nessuna duplicazione del codice del modello
- path configurabili via variabili d’ambiente (per Docker)

Variabili d’ambiente supportate:
- RUL_MODEL_PATH: path del checkpoint .pt
- RUL_SCALER_PATH: path dello scaler .joblib
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import joblib
import torch

from model_training.training.models.GNN_Transformer.st_gnn_transformer import STGNNTransformer


@dataclass
class InferenceBundle:
    model: torch.nn.Module
    scaler: object
    feature_cols: List[str]
    config: dict
    device: str


def _project_root() -> Path:
    # inference/ è nella root del progetto
    return Path(__file__).resolve().parent.parent


def default_model_path() -> Path:
    return _project_root() / "model_training" / "training" / "models" / "best_model.pt"


def default_scaler_path() -> Path:
    return _project_root() / "model_training" / "training" / "artifacts" / "scaler.joblib"


def load_inference_bundle(
    *,
    device: Optional[str] = None,
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
) -> InferenceBundle:
    """
    Carica checkpoint + scaler e restituisce un bundle pronto per EngineManager.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    mp = Path(model_path or os.environ.get("RUL_MODEL_PATH") or default_model_path()).resolve()
    sp = Path(scaler_path or os.environ.get("RUL_SCALER_PATH") or default_scaler_path()).resolve()

    checkpoint = torch.load(mp, map_location=device, weights_only=False)
    config = checkpoint.get("config")
    feature_cols = checkpoint.get("feature_cols")
    state_dict = checkpoint.get("model_state_dict")

    if config is None or feature_cols is None or state_dict is None:
        raise ValueError(
            "Checkpoint incompleto: servono 'config', 'feature_cols', 'model_state_dict'."
        )

    init_adj = torch.ones(len(feature_cols), len(feature_cols))
    model = STGNNTransformer(config, init_adj)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    scaler = joblib.load(sp)

    return InferenceBundle(
        model=model,
        scaler=scaler,
        feature_cols=list(feature_cols),
        config=config,
        device=str(device),
    )


