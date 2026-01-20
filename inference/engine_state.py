"""
Engine State Module

This module tracks the state of individual engines, including their
sensor history and predictions.
"""

from collections import deque
import numpy as np


class EngineState:
    """
    Tracks the state and history of a single engine.

    This class maintains a sliding window of recent sensor readings
    for each engine and stores the latest RUL prediction.

    Args:
        engine_id (int): Unique identifier for this engine
        window_size (int): Number of time steps to keep in memory
        num_features (int): Number of sensor features per time step
    """

    def __init__(self, engine_id, window_size, num_features):
        self.engine_id = engine_id
        self.window_size = window_size
        self.num_features = num_features

        # Use deque for efficient sliding window (automatically removes old data)
        self.buffer = deque(maxlen=window_size)

        # Track current cycle and latest prediction
        self.last_cycle = None
        self.last_prediction = None

    def add_event(self, event):
        """
        Add new sensor readings to the engine's history.

        Args:
            event (dict): Event containing sensor data with key 'features'

        Raises:
            ValueError: If feature dimensions don't match expected size
        """
        features = event["features"]

        # Verify feature size matches what we expect
        if features.shape[0] != self.num_features:
            raise ValueError("Feature size mismatch")

        # Add to sliding window (oldest data is automatically removed)
        self.buffer.append(features)

        # Update current cycle
        self.last_cycle = event["cycle"]

    def is_ready(self):
        """
        Check if we have enough history to make a prediction.

        Returns:
            bool: True if buffer is full, False otherwise
        """
        return len(self.buffer) == self.window_size

    def get_window(self):
        """
        Get the current sliding window of sensor data.

        Returns:
            np.ndarray: Array of shape (window_size, num_features)

        Raises:
            ValueError: If buffer is not full yet
        """
        if not self.is_ready():
            raise ValueError(f"Engine {self.engine_id} not ready")

        # Stack all time steps into a single array
        return np.stack(self.buffer, axis=0)

    def update_prediction(self, prediction):
        """
        Store the latest RUL prediction for this engine.

        Args:
            prediction (float): Predicted Remaining Useful Life in cycles
        """
        self.last_prediction = float(prediction)

    def reset(self):
        """
        Reset engine state to initial conditions.

        Clears all history and predictions.
        """
        self.buffer.clear()
        self.last_cycle = None
        self.last_prediction = None
