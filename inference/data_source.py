"""
Data Source Module

This module manages the sensor data for all engines in the simulation.
It reads data and provides it step by step to simulate real-time monitoring.
"""

import pandas as pd
import numpy as np

# Column names for the NASA C-MAPSS dataset
# id: engine identifier
# cycle: operational cycle number
# settings: engine operational settings
# sensors: 21 different sensor measurements
column_names = (
    ["id", "cycle"] +
    ["setting1", "setting2", "setting3"] +
    [f"sensor{i}" for i in range(1, 22)]
)


class DataSource:
    """
    Manages sensor data for multiple engines and simulates real-time data streaming.

    This class loads historical sensor data and provides it one cycle at a time
    for each engine, simulating a real-time monitoring system.

    Args:
        df (pd.DataFrame): DataFrame containing sensor data for all engines
        feature_cols (list): List of column names to use as features
    """

    def __init__(self, df, feature_cols):
        # Store which columns are used as features
        self.feature_cols = feature_cols

        # Split data by engine ID and store separately
        self.engines = {}
        for engine_id, g in df.groupby('id'):
            # Sort by cycle to ensure chronological order
            self.engines[engine_id] = g.sort_values('cycle').reset_index(drop=True)

        # Track current position for each engine (which cycle we're at)
        self.current_idx = {engine_id: 0 for engine_id in self.engines}

    def step(self):
        """
        Advance one time step and get sensor readings for all engines.

        Returns:
            list: List of events, one per engine. Each event is a dict with:
                - engine_id: Identifier of the engine
                - cycle: Current cycle number
                - features: Sensor readings as numpy array
        """
        events = []

        # Process each engine
        for engine_id, df_engine in self.engines.items():
            idx = self.current_idx[engine_id]

            # If we've reached the end of this engine's data, start over
            if idx >= len(df_engine):
                self.current_idx[engine_id] = 0
                idx = 0

            # Get sensor data for this cycle
            row = df_engine.iloc[idx]

            # Create event with engine data
            event = {
                "engine_id": engine_id,
                "cycle": int(row['cycle']),
                "features": row[self.feature_cols].values.astype(np.float32)
            }

            events.append(event)

            # Move to next cycle for this engine
            self.current_idx[engine_id] += 1

        return events

    def run(self, n_steps=1, verbose=True):
        """
        Run simulation for multiple steps.

        Args:
            n_steps (int): Number of time steps to simulate
            verbose (bool): Whether to print events
        """
        for t in range(n_steps):
            events = self.step()
            if verbose:
                for e in events:
                    print(e)
