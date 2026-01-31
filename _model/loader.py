"""
Loader

This module contains functions to load the trained model
and scaler from the artifacts folder.
"""

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
        model_path: Path to model file (optional, uses default)
        scaler_path: Path to scaler file (optional, uses default)
        device: Device to use (optional, auto-detects)

    Returns:
        Tuple with (model, scaler, config, feature_cols, device)
    """

    # Choose device (GPU if available, otherwise CPU)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use default paths if not provided
    artifacts = get_artifacts_path()
    if model_path is None:
        model_path = artifacts / "best_model.pt"
    if scaler_path is None:
        scaler_path = artifacts / "scaler.joblib"

    # Load the checkpoint file
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    feature_cols = checkpoint["feature_cols"]
    state_dict = checkpoint["model_state_dict"]

    # Create model with initial adjacency matrix
    num_sensors = len(feature_cols)
    init_adj = torch.ones(num_sensors, num_sensors)
    model = STGNNTransformer(config, init_adj)

    # Load trained weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load scaler
    scaler = joblib.load(scaler_path)

    return model, scaler, config, list(feature_cols), device
