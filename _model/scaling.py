"""
Scaling

This module provides the TimeSeriesScaler class for normalizing
3D time series data (batch, time_steps, features).
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class TimeSeriesScaler:
    """
    Scaler for 3D time series data.

    Uses sklearn StandardScaler internally, handling
    the reshape from 3D to 2D and back.
    """

    def __init__(self):
        """Create a new TimeSeriesScaler."""
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X):
        """
        Fit the scaler on training data.

        Args:
            X: numpy array of shape (batch, time_steps, features)
        """
        X_flat = X.reshape(-1, X.shape[-1])
        self.scaler.fit(X_flat)
        self.is_fitted = True
        return self

    def transform(self, X):
        """
        Transform data using the fitted scaler.

        Args:
            X: numpy array of shape (batch, time_steps, features)

        Returns:
            Scaled array with same shape as input
        """
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_flat)
        return X_scaled.reshape(original_shape)

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Reverse the transformation."""
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        X_original = self.scaler.inverse_transform(X_flat)
        return X_original.reshape(original_shape)
