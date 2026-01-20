"""
Scaling Module (Legacy Compatibility)

This module exists only to support loading the scaler.joblib file.
The actual implementation is now in model/scaling.py.
"""

# Import from new location for backward compatibility
from model.scaling import TimeSeriesScaler, StandardScaler

__all__ = ["TimeSeriesScaler", "StandardScaler"]
