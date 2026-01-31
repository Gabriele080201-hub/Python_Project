"""
Functions to load the model and scaler.

DO NOT use directly - use the Predictor class.
"""

import os
from pathlib import Path

import joblib
import torch

from .architecture import STGNNTransformer


def get_artifacts_path():
    """Return the path to the artifacts folder."""
    return Path(__file__).parent / "artifacts"


def load_model_and_scaler(model_path=None, scaler_path=None, device=None):
    """
    Load the model and scaler from files.

    Args:
        model_path: Path to model file (optional)
        scaler_path: Path to scaler file (optional)
        device: Device to use ('cpu' or 'cuda', optional)

    Returns:
        Tuple (model, scaler, config, feature_cols, device)
    """
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Default paths
    artifacts = get_artifacts_path()
    if model_path is None:
        model_path = artifacts / "best_model.pt"
    if scaler_path is None:
        scaler_path = artifacts / "scaler.joblib"

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    feature_cols = checkpoint["feature_cols"]
    state_dict = checkpoint["model_state_dict"]

    # Create model
    num_sensors = len(feature_cols)
    init_adj = torch.ones(num_sensors, num_sensors)
    model = STGNNTransformer(config, init_adj)

    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load scaler
    scaler = joblib.load(scaler_path)

    return model, scaler, config, list(feature_cols), device
