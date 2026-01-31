"""
DataSource

This is the module for simulating the data.

The class DataSource contains especially step(), used to 
simulate the incoming of one row of events for each engine
"""

import numpy as np
import pandas as pd


COLUMN_NAMES = (
    ["id", "cycle"] +
    ["setting1", "setting2", "setting3"] +
    [f"sensor{i}" for i in range(1, 22)]
)


class DataSource:
    """
    Data source for the simulation.

    The class reads the dataset and provides
    data step by step using step(), one cycle at a time for
    each engine.

    Attributes:
        feature_cols: list of columns used as input features.
    """


    def __init__(self, dataframe, feature_cols):
        """ 
        Args:
            dataframe: DataFrame of sensor data
            feature_cols: List of columns to use
        """
        self.feature_cols = feature_cols

        self.engines_data = {}

        for engine_id, group in dataframe.groupby("id"):
            sorted_data = group.sort_values("cycle").reset_index(drop=True)
            self.engines_data[engine_id] = sorted_data

        self.current_index = {eid: 0 for eid in self.engines_data}

    def step(self):
        """
        Move the simulation forward by one step.

        It returns a list of events, one for each engine, 
        with the following fields:
            - engine_id: Engine ID
            - cycle: Current cycle 
            - features: numpy array with sensor values
        """

        events = []

        for engine_id, df_engine in self.engines_data.items():
            idx = self.current_index[engine_id]

            # If we finished the data, restart from beginning otherwise
            # it would raise an error.
            if idx >= len(df_engine):
                self.current_index[engine_id] = 0
                idx = 0

            row = df_engine.iloc[idx]

            event = {
                "engine_id": engine_id,
                "cycle": int(row["cycle"]),
                "features": row[self.feature_cols].values.astype(np.float32)
            }
            events.append(event)

            self.current_index[engine_id] += 1

        return events