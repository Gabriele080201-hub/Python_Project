"""
Fleet module.

This module manages a group of engines and coordinates
the simulation using data from the DataSource.
"""

import pandas as pd

from .engine import Engine


class Fleet:
    """
    Manages a fleet of engines during the simulation.

    The Fleet receives data from the DataSource and
    sends it to the corresponding Engine objects.
    It also stores the results for later analysis.
    """


    def __init__(self, predictor, data_source):
        """
        Create a new Fleet.

        predictor: object used by engines to predict RUL
        data_source: object that provides data
        """

        self.predictor = predictor
        self.data_source = data_source

        # Dictionary that stores one Engine object for each engine_id
        self.engines = {}

        # Dictionary that stores the full history of each engine
        self.history = {}

        # Feature names, used later to build DataFrames
        self.feature_cols = data_source.feature_cols


    def step(self):
        """
        Run one step of the simulation.

        New data is read from the DataSource and sent
        to the corresponding engines.
        """

        # Get new data for all engines from the DataSource
        events = self.data_source.step()

        predictions = []

        # Process each engine event
        for event in events:
            engine_id = event["engine_id"]

            # Create engine if it does not exist
            if engine_id not in self.engines:
                self.engines[engine_id] = Engine(engine_id, self.predictor)
                self.history[engine_id] = []

            # Get engine and add reading (add_reading is a method of Engine)
            engine = self.engines[engine_id]
            engine.add_reading(event["features"])

            # Save to history the current state of the engine
            record = {
                "cycle": engine.cycle,
                "features": event["features"],
                "rul_prediction": engine.rul
            }
            self.history[engine_id].append(record)

            # If there is a prediction, adds it to results
            if engine.rul is not None:
                predictions.append({
                    "engine_id": engine_id,
                    "cycle": engine.cycle,
                    "rul_prediction": engine.rul
                })

        return predictions

    def get_status(self):
        """
        Return the current status of all engines.
        """

        status = []
        for engine_id, engine in self.engines.items():
            status.append({
                "engine_id": engine_id,
                "cycle": engine.cycle,
                "rul_prediction": engine.rul
            })
        return status

    def get_engine(self, engine_id):
        """
        Return an engine given its id
        """
        return self.engines.get(engine_id)

    def get_engine_ids(self):
        """
        Return the IDs of all active engines.
        """
        return sorted(self.engines.keys())

    def get_engine_history(self, engine_id):
        """
        Return the full history of one engine as a DataFrame.
        """

        if engine_id not in self.history:
            return pd.DataFrame()

        records = self.history[engine_id]
        if not records:
            return pd.DataFrame()

        # Build the DataFrame
        data = []
        for record in records:
            row = {
                "cycle": record["cycle"],
                "rul_prediction": record["rul_prediction"]
            }
            # Add sensor values
            for i, col_name in enumerate(self.feature_cols):
                row[col_name] = record["features"][i]
            data.append(row)

        return pd.DataFrame(data)
