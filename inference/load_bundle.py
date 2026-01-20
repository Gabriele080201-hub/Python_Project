"""
Model Loading Module

This module loads the trained model and scaler for inference.
Provides a simple interface for production use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import joblib
import torch

from model.GNN_Transformer.st_gnn_transformer import STGNNTransformer


@dataclass
class InferenceBundle:
    """
    Bundle containing everything needed for inference.

    Attributes:
        model: The trained PyTorch model
        scaler: Scaler for normalizing input data
        feature_cols: List of feature column names
        config: Model configuration dictionary
        device: Device the model is running on ('cpu' or 'cuda')
    """
    model: torch.nn.Module
    scaler: object
    feature_cols: List[str]
    config: dict
    device: str


def _project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path: Path to project root
    """
    # inference/ is in the root of the project
    return Path(__file__).resolve().parent.parent


def default_model_path() -> Path:
    """
    Get the default path to the model checkpoint.

    Returns:
        Path: Path to best_model.pt
    """
    return _project_root() / "artifacts" / "best_model.pt"


def default_scaler_path() -> Path:
    """
    Get the default path to the scaler.

    Returns:
        Path: Path to scaler.joblib
    """
    return _project_root() / "artifacts" / "scaler.joblib"


def load_inference_bundle(
    *,
    device: Optional[str] = None,
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
) -> InferenceBundle:
    """
    Load the model and scaler for inference.

    This function:
    1. Loads the model checkpoint
    2. Initializes the model architecture
    3. Loads the trained weights
    4. Loads the data scaler
    5. Returns everything in a convenient bundle

    Args:
        device: Device to run on ('cpu' or 'cuda'). Auto-detected if None.
        model_path: Path to model checkpoint. Uses default if None.
        scaler_path: Path to scaler file. Uses default if None.

    Returns:
        InferenceBundle: Bundle with model, scaler, and configuration

    Raises:
        ValueError: If checkpoint is missing required information

    Environment Variables:
        RUL_MODEL_PATH: Override default model path
        RUL_SCALER_PATH: Override default scaler path
    """
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine paths (priority: argument > environment > default)
    mp = Path(model_path or os.environ.get("RUL_MODEL_PATH") or default_model_path()).resolve()
    sp = Path(scaler_path or os.environ.get("RUL_SCALER_PATH") or default_scaler_path()).resolve()

    # Load model checkpoint
    checkpoint = torch.load(mp, map_location=device, weights_only=False)

    # Extract checkpoint components
    config = checkpoint.get("config")
    feature_cols = checkpoint.get("feature_cols")
    state_dict = checkpoint.get("model_state_dict")

    # Verify checkpoint has all required information
    if config is None or feature_cols is None or state_dict is None:
        raise ValueError(
            "Incomplete checkpoint: requires 'config', 'feature_cols', 'model_state_dict'."
        )

    # Initialize adjacency matrix (sensor relationships)
    # Start with all sensors connected
    init_adj = torch.ones(len(feature_cols), len(feature_cols))

    # Create model instance
    model = STGNNTransformer(config, init_adj)

    # Load trained weights
    model.load_state_dict(state_dict)

    # Move to specified device and set to evaluation mode
    model.to(device)
    model.eval()

    # Load the scaler
    scaler = joblib.load(sp)

    # Return complete bundle
    return InferenceBundle(
        model=model,
        scaler=scaler,
        feature_cols=list(feature_cols),
        config=config,
        device=str(device),
    )
