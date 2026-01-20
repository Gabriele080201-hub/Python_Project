"""
Preprocessing Package (Legacy Compatibility)

This package exists only to support loading the scaler.joblib file
that was saved with references to the old 'preprocessing' module.

The actual implementation is now in model/scaling.py.
"""

# Import from new location for backward compatibility
from model.scaling import TimeSeriesScaler, StandardScaler

__all__ = ["TimeSeriesScaler", "StandardScaler"]
