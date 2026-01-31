"""
Predictor

This module provides the interface to the ML model for RUL prediction.

The Predictor class hides all the complexity of the neural network,
offering a simple predict() method.

Usage:
    from predictor import Predictor

    predictor = Predictor()
    rul = predictor.predict(sensor_window)
"""

import numpy as np
import torch

from _model.loader import load_model_and_scaler


class Predictor:
    """
    Predicts the Remaining Useful Life (RUL) of an engine.

    This class wraps the ML model and provides a simple interface:
    give sensor data, get a RUL prediction.

    Attributes:
        window_size: Number of timesteps required (30)
        num_sensors: Number of sensors required (14)
        feature_cols: Names of sensor columns
    """

    def __init__(self, model_path=None, scaler_path=None):
        """
        Initialize the Predictor by loading the model.

        Args:
            model_path: Path to model file (optional, uses default)
            scaler_path: Path to scaler file (optional, uses default)
        """
        # Load model and scaler from _model/ folder
        result = load_model_and_scaler(model_path, scaler_path)
        self.model = result[0]
        self.scaler = result[1]
        self.config = result[2]
        self.feature_cols_list = result[3]
        self.device = result[4]

    @property
    def window_size(self):
        """Return the number of timesteps required for a prediction."""
        return self.config["window_size"]

    @property
    def num_sensors(self):
        """Return the number of sensors required."""
        return len(self.feature_cols_list)

    @property
    def feature_cols(self):
        """Return the names of sensor columns."""
        return self.feature_cols_list.copy()

    def predict(self, window):
        """
        Predict RUL from a window of sensor data.

        Args:
            window: Numpy array of shape (window_size, num_sensors)
                    Contains data from 30 timesteps for 14 sensors

        Returns:
            Predicted RUL in cycles (float)
        """
        # Check data shape
        window = np.array(window, dtype=np.float32)

        if window.shape != (self.window_size, self.num_sensors):
            raise ValueError(
                f"Wrong shape: expected ({self.window_size}, {self.num_sensors}), "
                f"got {window.shape}"
            )

        # Add batch dimension: (30, 14) -> (1, 30, 14)
        window = window[np.newaxis, :, :]

        # Normalize the data
        window = self.scaler.transform(window)

        # Remove batch dimension: (1, 30, 14) -> (30, 14)
        window = window[0]

        # Convert to PyTorch tensor
        # Shape required by model: (batch, time, sensors, features)
        tensor = torch.tensor(window, dtype=torch.float32)
        tensor = tensor.unsqueeze(0)  # Add batch
        tensor = tensor.unsqueeze(-1)  # Add features
        tensor = tensor.to(self.device)

        # Run prediction (no gradient computation)
        with torch.no_grad():
            prediction = self.model(tensor)

        # Return value as Python float
        return float(prediction.item())
