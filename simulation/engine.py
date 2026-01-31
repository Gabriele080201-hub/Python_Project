"""
Engine

This module manages a single engine in the simulation.

The Engine class keeps a buffer of sensor readings and
computes RUL predictions when enough data is available.
"""

from collections import deque
import numpy as np


class Engine:
    """
    Represents a single engine with its data history.

    Each Engine keeps a sliding window buffer of the latest
    sensor readings. When the buffer is full, it can
    compute a RUL prediction.

    Attributes:
        engine_id: Unique engine identifier
        cycle: Current operating cycle
        rul: Latest RUL prediction (None if not available)
    """

    def __init__(self, engine_id, predictor):
        """
        Initialize a new Engine.

        Args:
            engine_id: Unique engine identifier
            predictor: Predictor object for predictions
        """
        self.engine_id = engine_id
        self.cycle = 0
        self.rul = None

        self.predictor = predictor
        self.buffer = deque(maxlen=predictor.window_size)
        self.history = []  # List of tuples (cycle, rul)

    def add_reading(self, sensor_data):
        """
        Add a sensor reading and update the prediction.

        When the buffer reaches the required size (30 timesteps),
        a new RUL prediction is automatically computed.

        Args:
            sensor_data: Numpy array with sensor values
        """
        self.cycle += 1
        self.buffer.append(sensor_data)

        # If buffer is full, compute prediction
        if self.is_ready():
            window = np.array(self.buffer)
            self.rul = self.predictor.predict(window)
            self.history.append((self.cycle, self.rul))

    def is_ready(self):
        """
        Check if the engine has enough data for a prediction.

        Returns:
            True if buffer is full, False otherwise
        """
        return len(self.buffer) == self.predictor.window_size

    def get_history(self):
        """
        Return the history of RUL predictions.

        Returns:
            List of tuples (cycle, rul) with all predictions
        """
        return self.history.copy()
