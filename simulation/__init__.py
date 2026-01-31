"""
Simulation Package - Classes for fleet simulation.

This package contains OOP classes to simulate
the monitoring of an aircraft engine fleet.

Classes:
    Engine: Manages a single engine
    Fleet: Manages the entire fleet
    DataSource: Provides sensor data

Example:
    from simulation import Fleet, DataSource
    from predictor import Predictor

    predictor = Predictor()
    data_source = DataSource(dataframe, feature_cols)
    fleet = Fleet(predictor, data_source)

    fleet.step()  # Advance simulation
    status = fleet.get_status()  # Engine status
"""

from .engine import Engine
from .fleet import Fleet
from .data_source import DataSource, COLUMN_NAMES

__all__ = ["Engine", "Fleet", "DataSource", "COLUMN_NAMES"]
