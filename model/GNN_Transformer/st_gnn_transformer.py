"""
Spatial-Temporal GNN Transformer Model

This is the main model that combines Graph Neural Networks (GNN)
and Transformers to predict Remaining Useful Life (RUL) of engines.
"""

import torch
import torch.nn as nn

from .layers import GraphConvolutionLayer
from .positional_encoding import PositionalEncoding


class STGNNTransformer(nn.Module):
    """
    Spatial-Temporal Graph Neural Network + Transformer for RUL prediction.

    This model processes sensor data in two stages:
    1. Spatial Processing (GNN): Learns relationships between different sensors
    2. Temporal Processing (Transformer): Learns patterns over time

    The combination allows the model to understand both:
    - Which sensors are related to each other (spatial)
    - How sensor values change over time (temporal)

    Args:
        config (dict): Configuration dictionary containing:
            - num_nodes: Number of sensors
            - gnn_hidden_dim: Hidden dimension for GNN layer
            - trans_d_model: Dimension of transformer embeddings
            - input_features: Number of input features per sensor
            - trans_nhead: Number of attention heads in transformer
            - trans_layers: Number of transformer encoder layers
            - dropout_prob: Dropout probability
        init_adj_matrix (torch.Tensor): Initial adjacency matrix showing sensor relationships
    """

    def __init__(self, config, init_adj_matrix):
        super().__init__()

        # Save configuration
        self.num_nodes = config["num_nodes"]
        H_g = config["gnn_hidden_dim"]
        d_model = config["trans_d_model"]

        # Learnable adjacency matrix (sensor relationships)
        # Starts from initial matrix but can be updated during training
        self.adj_parameter = nn.Parameter(init_adj_matrix.float())

        # Graph Convolution Layer (processes spatial relationships)
        self.gcn = GraphConvolutionLayer(
            input_dim=config["input_features"],
            output_dim=H_g
        )

        # Linear layer to convert GNN output to transformer input size
        self.flatten_dim = self.num_nodes * H_g
        self.embedding = nn.Linear(self.flatten_dim, d_model)

        # Positional encoding (adds time step information)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer Encoder Layer configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config["trans_nhead"],
            dim_feedforward=d_model,
            dropout=config["dropout_prob"],
            activation="gelu",
            batch_first=True
        )

        # Stack multiple transformer layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["trans_layers"]
        )

        # Final regression layer (outputs RUL prediction)
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def get_normalized_adj(self):
        """
        Normalize the adjacency matrix for stable graph convolutions.

        This ensures that aggregating information from neighbors doesn't
        lead to exploding values. Uses symmetric normalization:
        A_norm = D^(-1/2) * A * D^(-1/2)

        Returns:
            torch.Tensor: Normalized adjacency matrix
        """
        # Make the adjacency matrix symmetric
        A = 0.5 * (self.adj_parameter + self.adj_parameter.t())

        # Add self-loops (each sensor also considers its own value)
        A = A + torch.eye(A.size(0), device=A.device)

        # Calculate degree matrix (sum of connections for each node)
        deg = torch.sum(torch.abs(A), dim=1)

        # Calculate D^(-1/2) for normalization
        D_inv_sqrt = torch.diag(torch.pow(deg + 1e-6, -0.5))

        # Apply symmetric normalization
        return D_inv_sqrt @ A @ D_inv_sqrt

    def forward(self, x):
        """
        Forward pass to predict RUL from sensor data.

        Args:
            x (torch.Tensor): Input sensor data
                Shape: (batch_size, time_steps, num_sensors, features)

        Returns:
            torch.Tensor: Predicted RUL values, shape (batch_size, 1)
        """
        # Get input dimensions
        B, T, N, F = x.shape  # Batch, Time, Nodes (sensors), Features

        # Get normalized adjacency matrix
        A = self.get_normalized_adj()

        # Step 1: Process spatial relationships with GNN
        # Reshape to process all time steps together
        x = x.reshape(B * T, N, F)
        gcn_out = self.gcn(x, A)

        # Step 2: Prepare sequence for transformer
        # Reshape back to separate time steps
        seq = gcn_out.reshape(B, T, -1)

        # Step 3: Convert to transformer embedding dimension
        seq = self.embedding(seq)

        # Step 4: Add positional encoding (time step information)
        seq = self.pos_encoder(seq)

        # Step 5: Process temporal patterns with transformer
        out = self.transformer_encoder(seq)

        # Step 6: Aggregate over time (average pooling)
        # This creates a single representation for the entire sequence
        final_state = out.mean(dim=1)

        # Step 7: Predict RUL value
        return self.regressor(final_state)
