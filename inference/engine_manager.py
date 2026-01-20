"""
Engine Manager Module

This module handles the inference process: preparing data,
running predictions, and managing multiple engine states.
"""

import torch
import numpy as np
from .engine_state import EngineState


class EngineManager:
    """
    Manages inference for multiple engines.

    This class coordinates the prediction process:
    1. Maintains state for each engine
    2. Prepares sensor data (scaling and formatting)
    3. Runs the neural network model
    4. Returns RUL predictions

    Args:
        model (torch.nn.Module): The trained RUL prediction model
        scaler (object): Scaler for normalizing sensor data
        feature_cols (list): List of feature column names
        window_size (int): Number of time steps needed for prediction
        device (str): Device to run model on ('cpu' or 'cuda')
    """

    def __init__(self, model, scaler, feature_cols, window_size, device="cpu"):
        # Store model and set to evaluation mode (no training)
        self.model = model
        self.model.eval()

        # Store preprocessing components
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.device = device

        # Dictionary to track state of each engine
        self.engine_states = {}

    def process_events(self, events):
        """
        Process sensor readings and generate RUL predictions.

        Args:
            events (list): List of sensor events from all engines

        Returns:
            list: List of prediction dictionaries with keys:
                - engine_id: Engine identifier
                - cycle: Current cycle number
                - rul_prediction: Predicted remaining useful life
        """
        predictions = []

        for event in events:
            engine_id = event["engine_id"]

            # Create state for new engines
            if engine_id not in self.engine_states:
                self.engine_states[engine_id] = EngineState(
                    engine_id=engine_id,
                    window_size=self.window_size,
                    num_features=len(self.feature_cols)
                )

            # Get state for this engine
            engine_state = self.engine_states[engine_id]

            # Add new sensor reading
            engine_state.add_event(event)

            # Skip prediction if we don't have enough history yet
            if not engine_state.is_ready():
                continue

            # Step 1: Get sliding window of sensor data
            window = engine_state.get_window()  # Shape: (time_steps, features)

            # Step 2: Add batch dimension for scaler
            window = window[np.newaxis, :, :]  # Shape: (1, time_steps, features)

            # Step 3: Normalize using the scaler
            window = self.scaler.transform(window)  # Shape: (1, time_steps, features)

            # Step 4: Remove batch dimension
            window = window[0]  # Shape: (time_steps, features)

            # Step 5: Convert to PyTorch tensor and add required dimensions
            # Model expects: (batch, time_steps, num_sensors, features_per_sensor)
            window = torch.tensor(window, dtype=torch.float32) \
                .unsqueeze(0).unsqueeze(-1).to(self.device)

            # Step 6: Run inference (no gradient computation needed)
            with torch.no_grad():
                pred = self.model(window).item()

            # Step 7: Store prediction in engine state
            engine_state.update_prediction(pred)

            # Step 8: Add to results
            predictions.append({
                "engine_id": engine_id,
                "cycle": engine_state.last_cycle,
                "rul_prediction": pred
            })

        return predictions

    def reset_engine(self, engine_id):
        """
        Reset a specific engine's state.

        Args:
            engine_id (int): ID of engine to reset
        """
        if engine_id in self.engine_states:
            self.engine_states[engine_id].reset()
