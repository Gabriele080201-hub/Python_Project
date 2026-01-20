"""
Fleet Controller Module

This module coordinates the entire fleet monitoring system,
managing data flow and maintaining history for all engines.
"""

from collections import defaultdict
from typing import List, Dict, Any
import pandas as pd
import numpy as np


class FleetController:
    """
    Orchestrates the fleet monitoring system.

    This class manages the complete data flow:
    1. Gets sensor data from the data source
    2. Sends it to the engine manager for predictions
    3. Maintains historical data for visualization

    Args:
        data_source (DataSource): Source of sensor data
        engine_manager (EngineManager): Manager for running predictions
    """

    def __init__(self, data_source, engine_manager):
        self.data_source = data_source
        self.engine_manager = engine_manager

        # Store feature column names for creating DataFrames
        self.feature_cols = data_source.feature_cols

        # Store complete history for each engine
        # history[engine_id] = [{cycle, features, rul_prediction}, ...]
        self.history = defaultdict(list)

        # Track simulation progress
        self.current_step = 0

    def step(self) -> List[Dict[str, Any]]:
        """
        Execute one simulation step.

        This method:
        1. Gets new sensor data for all engines
        2. Generates RUL predictions
        3. Updates internal history

        Returns:
            list: List of prediction dictionaries
        """
        # Step 1: Get new sensor readings from all engines
        events = self.data_source.step()

        # Step 2: Generate predictions
        predictions = self.engine_manager.process_events(events)

        # Step 3: Create a quick lookup map for predictions
        # Note: predictions only exist if the engine has enough history
        pred_map = {p['engine_id']: p['rul_prediction'] for p in predictions}

        # Step 4: Update history for each engine
        for event in events:
            eid = event['engine_id']

            record = {
                "cycle": event['cycle'],
                "features": event['features'],  # Numpy array of sensor values
                "rul_prediction": pred_map.get(eid, None)  # None if not ready yet
            }
            self.history[eid].append(record)

        # Step 5: Increment step counter
        self.current_step += 1

        return predictions

    def get_fleet_table(self) -> List[Dict[str, Any]]:
        """
        Get the current status of all engines.

        Returns:
            list: List of dictionaries with the latest state of each engine.
                Each dict contains: engine_id, cycle, rul_prediction
        """
        table = []
        for eid, records in self.history.items():
            if not records:
                continue

            # Get most recent record
            last = records[-1]
            table.append({
                "engine_id": eid,
                "cycle": last['cycle'],
                "rul_prediction": last['rul_prediction']
            })
        return table

    def get_engine_history_df(self, engine_id: int) -> pd.DataFrame:
        """
        Convert a single engine's history to a Pandas DataFrame.

        This creates a table with:
        - Cycle number
        - RUL prediction
        - All sensor values (one column per sensor)

        Args:
            engine_id (int): ID of the engine to get history for

        Returns:
            pd.DataFrame: DataFrame containing the engine's complete history
        """
        if engine_id not in self.history or not self.history[engine_id]:
            return pd.DataFrame()

        records = self.history[engine_id]

        expanded_data = []
        for r in records:
            # Create base dictionary with cycle and RUL
            row = {
                "cycle": r["cycle"],
                "rul_prediction": r["rul_prediction"]
            }

            # Expand features array into separate columns
            # Example: {"sensor1": 591.2, "sensor2": 1.3, ...}
            sensor_values = dict(zip(self.feature_cols, r["features"]))
            row.update(sensor_values)

            expanded_data.append(row)

        return pd.DataFrame(expanded_data)

    def reset(self):
        """
        Reset the entire simulation to initial state.

        Clears all history and engine states.
        """
        self.history.clear()
        self.engine_manager.engine_states.clear()
        self.current_step = 0
