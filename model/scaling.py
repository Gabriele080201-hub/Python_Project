"""
Scaling Module

Time series scaler for normalizing 3D data (batch, time_steps, features).
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class TimeSeriesScaler:
    """
    Scaler for 3D time series data.

    Normalizes data of shape (batch, time_steps, features) by flattening,
    scaling with StandardScaler, and reshaping back.
    """

    def __init__(self):
        """Initialize with a StandardScaler."""
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X):
        """
        Fit the scaler on training data.

        Args:
            X: Array of shape (batch, time_steps, features).

        Returns:
            self: The fitted scaler.
        """
        X_reshaped = X.reshape(-1, X.shape[-1])
        self.scaler.fit(X_reshaped)
        self.is_fitted = True
        return self

    def transform(self, X):
        """
        Transform data using the fitted scaler.

        Args:
            X: Array of shape (batch, time_steps, features).

        Returns:
            Scaled array with same shape as input.
        """
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_reshaped)
        return X_scaled.reshape(original_shape)

    def fit_transform(self, X):
        """
        Fit and transform in one step.

        Args:
            X: Array of shape (batch, time_steps, features).

        Returns:
            Scaled array with same shape as input.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Reverse the scaling transformation.

        Args:
            X: Scaled array of shape (batch, time_steps, features).

        Returns:
            Original scale array with same shape as input.
        """
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_original = self.scaler.inverse_transform(X_reshaped)
        return X_original.reshape(original_shape)


__all__ = ["TimeSeriesScaler", "StandardScaler"]
