"""
Scaler for time series data.

DO NOT use directly - use the Predictor class.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class TimeSeriesScaler:
    """
    Scaler for 3D data (batch, time_steps, features).

    Normalizes data using sklearn StandardScaler,
    automatically handling the required reshape.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X):
        """Fit the scaler on training data."""
        X_reshaped = X.reshape(-1, X.shape[-1])
        self.scaler.fit(X_reshaped)
        self.is_fitted = True
        return self

    def transform(self, X):
        """Transform data using the fitted scaler."""
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_reshaped)
        return X_scaled.reshape(original_shape)

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Reverse the transformation."""
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_original = self.scaler.inverse_transform(X_reshaped)
        return X_original.reshape(original_shape)
