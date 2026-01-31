"""
Predictor Module - API for RUL predictions.

This module provides a simple interface to predict the
Remaining Useful Life (RUL) of aircraft engines.

Usage example:
    from predictor import Predictor

    predictor = Predictor()
    rul = predictor.predict(sensor_window)
"""

import numpy as np
import torch

from _model.loader import load_model_and_scaler


class Predictor:
    """
    Class to predict the Remaining Useful Life (RUL) of an engine.

    This class hides all the ML model complexity,
    offering a simple interface: give data, get prediction.

    Attributes:
        window_size (int): Number of timesteps required (30)
        num_sensors (int): Number of sensors required (14)
        feature_cols (list): Names of sensor columns

    Example:
        >>> predictor = Predictor()
        >>> # window has shape (30, 14) - 30 timesteps, 14 sensors
        >>> rul = predictor.predict(window)
        >>> print(f"Predicted RUL: {rul:.1f} cycles")
    """

    def __init__(self, model_path=None, scaler_path=None):
        """
        Initialize the Predictor by loading the model.

        Args:
            model_path: Path to model file (optional, uses default)
            scaler_path: Path to scaler file (optional, uses default)
        """
        # Load model and scaler (details hidden in _model/)
        result = load_model_and_scaler(model_path, scaler_path)
        self._model = result[0]
        self._scaler = result[1]
        self._config = result[2]
        self._feature_cols = result[3]
        self._device = result[4]

    @property
    def window_size(self):
        """Return the number of timesteps required for a prediction."""
        return self._config["window_size"]

    @property
    def num_sensors(self):
        """Return the number of sensors required."""
        return len(self._feature_cols)

    @property
    def feature_cols(self):
        """Return the names of sensor columns."""
        return self._feature_cols.copy()

    def predict(self, window):
        """
        Predict RUL from a window of sensor data.

        Args:
            window: Numpy array of shape (window_size, num_sensors)
                    Contains data from 30 timesteps for 14 sensors

        Returns:
            float: Predicted RUL in cycles

        Raises:
            ValueError: If data shape is incorrect

        Example:
            >>> window = np.random.randn(30, 14)  # example data
            >>> rul = predictor.predict(window)
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
        window = self._scaler.transform(window)

        # Remove batch dimension: (1, 30, 14) -> (30, 14)
        window = window[0]

        # Convert to PyTorch tensor
        # Shape required by model: (batch, time, sensors, features)
        tensor = torch.tensor(window, dtype=torch.float32)
        tensor = tensor.unsqueeze(0)  # Add batch
        tensor = tensor.unsqueeze(-1)  # Add features
        tensor = tensor.to(self._device)

        # Run prediction (no gradient computation)
        with torch.no_grad():
            prediction = self._model(tensor)

        # Return value as Python float
        return float(prediction.item())
