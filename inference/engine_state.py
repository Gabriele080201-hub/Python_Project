from collections import deque
import numpy as np

class EngineState:
    def __init__(self, engine_id, window_size, num_features):
        self.engine_id = engine_id
        self.window_size = window_size
        self.num_features = num_features

        self.buffer = deque(maxlen=window_size)
        self.last_cycle = None
        self.last_prediction = None

    def add_event(self, event):
        features = event["features"]
        if features.shape[0] != self.num_features:
            raise ValueError("Feature size mismatch")

        self.buffer.append(features)
        self.last_cycle = event["cycle"]

    def is_ready(self):
        return len(self.buffer) == self.window_size

    def get_window(self):
        if not self.is_ready():
            raise ValueError(f"Engine {self.engine_id} not ready")
        return np.stack(self.buffer, axis=0)

    def update_prediction(self, prediction):
        self.last_prediction = float(prediction)

    def reset(self):
        self.buffer.clear()
        self.last_cycle = None
        self.last_prediction = None