"""
Fleet Module - Engine fleet management.

Contains the Fleet class that orchestrates the simulation
of a complete engine fleet.
"""

import pandas as pd

from .engine import Engine


class Fleet:
    """
    Manages an engine fleet and their RUL predictions.

    Fleet is the central point of the simulation: it coordinates
    data reading, single engine management, and keeps
    history for visualization.

    Attributes:
        feature_cols (list): Names of sensor columns

    Example:
        >>> fleet = Fleet(predictor, data_source)
        >>> fleet.step()  # Advance simulation
        >>> status = fleet.get_status()  # Status of all engines
    """

    def __init__(self, predictor, data_source):
        """
        Initialize the Fleet.

        Args:
            predictor (Predictor): Object for RUL predictions
            data_source (DataSource): Source of sensor data
        """
        self._predictor = predictor
        self._data_source = data_source
        self._engines = {}  # Dict[int, Engine]
        self._history = {}  # Dict[int, list] - complete history

        # Expose feature columns for the UI
        self.feature_cols = data_source.feature_cols

    def step(self):
        """
        Execute one simulation step.

        Reads new data from DataSource, passes it to the
        corresponding engines, and updates the history.

        Returns:
            list: List of new predictions (only ready engines)
        """
        # Read new events
        events = self._data_source.step()

        predictions = []

        for event in events:
            engine_id = event["engine_id"]

            # Create engine if it does not exist
            if engine_id not in self._engines:
                self._engines[engine_id] = Engine(engine_id, self._predictor)
                self._history[engine_id] = []

            # Get engine and add reading
            engine = self._engines[engine_id]
            engine.add_reading(event["features"])

            # Save to history
            record = {
                "cycle": engine.cycle,
                "features": event["features"],
                "rul_prediction": engine.rul  # None if not ready
            }
            self._history[engine_id].append(record)

            # If there is a prediction, add it to results
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

        Returns:
            list: List of dicts with engine_id, cycle, rul_prediction
        """
        status = []
        for engine_id, engine in self._engines.items():
            status.append({
                "engine_id": engine_id,
                "cycle": engine.cycle,
                "rul_prediction": engine.rul
            })
        return status

    def get_engine(self, engine_id):
        """
        Return a specific engine.

        Args:
            engine_id (int): Engine ID

        Returns:
            Engine or None: The requested engine or None if not found
        """
        return self._engines.get(engine_id)

    def get_engine_ids(self):
        """
        Return the list of active engine IDs.

        Returns:
            list: Sorted list of IDs
        """
        return sorted(self._engines.keys())

    def get_engine_history(self, engine_id):
        """
        Return an engine's history as a DataFrame.

        Args:
            engine_id (int): Engine ID

        Returns:
            pd.DataFrame: DataFrame with cycle, rul_prediction, and sensors
        """
        if engine_id not in self._history:
            return pd.DataFrame()

        records = self._history[engine_id]
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

    def reset(self):
        """Reset the simulation."""
        self._engines.clear()
        self._history.clear()
        self._data_source.reset()
