"""
Engine module.

This module defines the Engine class, which represents
a single engine in the simulation.
"""

from collections import deque
import numpy as np


class Engine:
    """
    Represents a single engine in the simulation.

    The engine receives sensor data over time and computes
    a RUL prediction when enough data is available.
    """

    def __init__(self, engine_id, predictor):
        """
        Create a new Engine.

        engine_id: ID of the engine
        predictor: object used to compute RUL predictions
        """
        self.engine_id = engine_id
        self.cycle = 0
        self.rul = None

        self.predictor = predictor

        # Buffer that stores the latest sensor readings
        self.buffer = deque(maxlen=predictor.window_size)

        self.history = []

    def add_reading(self, sensor_data):
        """
        Add new sensor data to the engine.

        When enough data is available, a new RUL prediction
        is computed automatically.
        """
        # Move to the next cycle
        self.cycle += 1

        # Store the new sensor data
        self.buffer.append(sensor_data)

        # If the buffer is full, compute a prediction
        if self.is_ready():
            window = np.array(self.buffer)
            self.rul = self.predictor.predict(window)
            self.history.append((self.cycle, self.rul))

    def is_ready(self):
        """
        Return True if the engine is ready to predict RUL.
        """
        return len(self.buffer) == self.predictor.window_size

    def get_history(self):
        """
        Return the history of RUL predictions.
        """
        return self.history.copy()
