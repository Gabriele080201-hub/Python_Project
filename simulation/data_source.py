"""
DataSource Module - Data source for the simulation.

Contains the DataSource class that reads dataset data
and provides it step by step, simulating real-time streaming.
"""

import numpy as np
import pandas as pd


# Column names for the NASA C-MAPSS dataset
COLUMN_NAMES = (
    ["id", "cycle"] +
    ["setting1", "setting2", "setting3"] +
    [f"sensor{i}" for i in range(1, 22)]
)


class DataSource:
    """
    Data source for engine monitoring simulation.

    Reads data from the NASA C-MAPSS dataset and provides it
    one cycle at a time for each engine, simulating
    real-time data streaming.

    Attributes:
        feature_cols (list): List of column names used as features

    Example:
        >>> source = DataSource(dataframe, feature_cols)
        >>> events = source.step()  # Get data for all engines
    """

    def __init__(self, dataframe, feature_cols):
        """
        Initialize the DataSource.

        Args:
            dataframe (pd.DataFrame): DataFrame with sensor data
            feature_cols (list): List of columns to use as features
        """
        self.feature_cols = feature_cols

        # Separate data for each engine
        self._engines_data = {}
        for engine_id, group in dataframe.groupby("id"):
            # Sort by cycle and reset index
            sorted_data = group.sort_values("cycle").reset_index(drop=True)
            self._engines_data[engine_id] = sorted_data

        # Track current position for each engine
        self._current_index = {eid: 0 for eid in self._engines_data}

    def step(self):
        """
        Advance one step and return data for all engines.

        Returns:
            list: List of dicts, one per engine, with:
                - engine_id: Engine ID
                - cycle: Current cycle number
                - features: Numpy array with sensor values
        """
        events = []

        for engine_id, df_engine in self._engines_data.items():
            idx = self._current_index[engine_id]

            # If we finished the data, restart from beginning
            if idx >= len(df_engine):
                self._current_index[engine_id] = 0
                idx = 0

            # Read current row
            row = df_engine.iloc[idx]

            # Create event with data
            event = {
                "engine_id": engine_id,
                "cycle": int(row["cycle"]),
                "features": row[self.feature_cols].values.astype(np.float32)
            }
            events.append(event)

            # Advance to next index
            self._current_index[engine_id] += 1

        return events

    def reset(self):
        """Reset the data source to the beginning."""
        self._current_index = {eid: 0 for eid in self._engines_data}

    def get_engine_ids(self):
        """
        Return the list of available engine IDs.

        Returns:
            list: List of engine IDs
        """
        return list(self._engines_data.keys())
