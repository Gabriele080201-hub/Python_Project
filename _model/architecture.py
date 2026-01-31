"""
GNN-Transformer model architecture.

This module contains the neural network model that combines:
- Graph Neural Network (GNN) for spatial relationships between sensors
- Transformer for temporal patterns

DO NOT use directly - use the Predictor class.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# GRAPH CONVOLUTION LAYER
# =============================================================================

class GraphConvolutionLayer(nn.Module):
    """
    Graph convolution layer.

    Processes sensor data as graph nodes,
    applying: H = GELU(LayerNorm(A @ X @ W + b))
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, adj):
        support = torch.matmul(adj, x)
        out = torch.matmul(support, self.weight) + self.bias
        out = self.norm(out)
        return F.gelu(out)


# =============================================================================
# POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the Transformer.

    Adds temporal position information using
    sine and cosine functions.
    """

    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# =============================================================================
# MAIN MODEL
# =============================================================================

class STGNNTransformer(nn.Module):
    """
    Spatial-Temporal GNN + Transformer model for RUL prediction.

    Combines:
    1. GNN for spatial relationships between sensors
    2. Transformer for temporal patterns
    """

    def __init__(self, config, init_adj_matrix):
        super().__init__()

        self.num_nodes = config["num_nodes"]
        H_g = config["gnn_hidden_dim"]
        d_model = config["trans_d_model"]

        # Learnable adjacency matrix
        self.adj_parameter = nn.Parameter(init_adj_matrix.float())

        # GNN layer
        self.gcn = GraphConvolutionLayer(
            input_dim=config["input_features"],
            output_dim=H_g
        )

        # Embedding for transformer
        self.flatten_dim = self.num_nodes * H_g
        self.embedding = nn.Linear(self.flatten_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config["trans_nhead"],
            dim_feedforward=d_model,
            dropout=config["dropout_prob"],
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["trans_layers"]
        )

        # Final regression layer
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def get_normalized_adj(self):
        """Normalize the adjacency matrix."""
        A = 0.5 * (self.adj_parameter + self.adj_parameter.t())
        A = A + torch.eye(A.size(0), device=A.device)
        deg = torch.sum(torch.abs(A), dim=1)
        D_inv_sqrt = torch.diag(torch.pow(deg + 1e-6, -0.5))
        return D_inv_sqrt @ A @ D_inv_sqrt

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, time_steps, num_sensors, features)

        Returns:
            Tensor of shape (batch, 1) with RUL prediction
        """
        B, T, N, F = x.shape
        A = self.get_normalized_adj()

        # GNN: process spatial relationships
        x = x.reshape(B * T, N, F)
        gcn_out = self.gcn(x, A)

        # Prepare sequence for transformer
        seq = gcn_out.reshape(B, T, -1)
        seq = self.embedding(seq)
        seq = self.pos_encoder(seq)

        # Transformer: process temporal patterns
        out = self.transformer_encoder(seq)

        # Temporal pooling and prediction
        final_state = out.mean(dim=1)
        return self.regressor(final_state)
