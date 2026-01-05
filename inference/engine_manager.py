import torch
import numpy as np
from .engine_state import EngineState

class EngineManager:
    def __init__(self, model, scaler, feature_cols, window_size, device="cpu"):
        self.model = model
        self.model.eval()

        self.scaler = scaler
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.device = device

        self.engine_states = {}

    def process_events(self, events):
        predictions = []

        for event in events:
            engine_id = event["engine_id"]

            if engine_id not in self.engine_states:
                self.engine_states[engine_id] = EngineState(
                    engine_id=engine_id,
                    window_size=self.window_size,
                    num_features=len(self.feature_cols)
                )

            engine_state = self.engine_states[engine_id]
            engine_state.add_event(event)

            if not engine_state.is_ready():
                continue

            window = engine_state.get_window()          # (T, F)
            window = window[np.newaxis, :, :]           # (1, T, F)
            window = self.scaler.transform(window)      # (1, T, F)
            window = window[0]                          # (T, F) → opzionale

            window = torch.tensor(window, dtype=torch.float32) \
                .unsqueeze(0).unsqueeze(-1).to(self.device)

            with torch.no_grad():
                pred = self.model(window).item()

            engine_state.update_prediction(pred)

            predictions.append({
                "engine_id": engine_id,
                "cycle": engine_state.last_cycle,
                "rul_prediction": pred
            })

        return predictions
    
    def reset_engine(self, engine_id):
        if engine_id in self.engine_states:
            self.engine_states[engine_id].reset()