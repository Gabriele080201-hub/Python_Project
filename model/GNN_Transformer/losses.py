"""
Loss Functions Module

This module contains custom loss functions for model training.
"""

import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    """
    Root Mean Squared Error (RMSE) Loss Function.

    RMSE is commonly used for regression problems like predicting
    Remaining Useful Life (RUL). It measures the average magnitude
    of prediction errors.

    RMSE = sqrt(mean((predictions - actual)^2))

    Args:
        eps (float): Small value to prevent division by zero (default: 1e-6)
    """

    def __init__(self, eps=1e-6):
        super().__init__()

        # Use built-in Mean Squared Error loss
        self.mse = nn.MSELoss()

        # Small epsilon value for numerical stability
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Calculate RMSE loss.

        Args:
            y_pred (torch.Tensor): Predicted values
            y_true (torch.Tensor): Actual values

        Returns:
            torch.Tensor: RMSE loss value (scalar)
        """
        # Calculate MSE and then take square root
        # Add epsilon to avoid sqrt(0)
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)
