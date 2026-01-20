"""
Positional Encoding Module

This module adds positional information to sequence data so the model
knows the order of time steps.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer models.

    Since Transformers don't naturally understand sequence order,
    this layer adds position information using sine and cosine functions.
    This helps the model understand that time step 1 comes before time step 2, etc.

    The encoding is based on the original "Attention Is All You Need" paper.

    Args:
        d_model (int): Dimension of the model embeddings
        max_len (int): Maximum sequence length to support (default: 500)
    """

    def __init__(self, d_model, max_len=500):
        super().__init__()

        # Create a matrix to store positional encodings
        pe = torch.zeros(max_len, d_model)

        # Create position indices [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Calculate the division term for the sinusoidal functions
        # This creates different frequencies for each dimension
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices in the encoding
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices in the encoding
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, won't be trained)
        # Add batch dimension
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        Add positional encoding to input.

        Args:
            x (torch.Tensor): Input tensor, shape (batch, sequence_length, d_model)

        Returns:
            torch.Tensor: Input with positional information added, same shape as input
        """
        # Add positional encoding to input
        # Only use the portion that matches the sequence length
        return x + self.pe[:, :x.size(1)]
