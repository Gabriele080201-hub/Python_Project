"""
Architecture

This module contains the neural network model for RUL prediction.
The model combines a Graph Neural Network (GNN) with a Transformer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolutionLayer(nn.Module):
    """
    Graph convolution layer.

    Processes sensor data treating sensors as nodes in a graph.
    Each node aggregates information from its neighbors.
    """

    def __init__(self, input_dim, output_dim):
        """
        Create a new GraphConvolutionLayer.

        Args:
            input_dim: Size of input features
            output_dim: Size of output features
        """
        super().__init__()

        # Learnable weight matrix and bias
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))

        # Initialize weights
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

        # Layer normalization
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, adj):
        """
        Forward pass.

        Args:
            x: Input tensor (batch, nodes, features)
            adj: Adjacency matrix (nodes, nodes)

        Returns:
            Output tensor (batch, nodes, output_dim)
        """
        # Aggregate neighbor information
        support = torch.matmul(adj, x)

        # Apply linear transformation
        out = torch.matmul(support, self.weight) + self.bias

        # Normalize and activate
        out = self.norm(out)
        return F.gelu(out)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the Transformer.

    Adds information about the position of each time step
    using sine and cosine functions.
    """

    def __init__(self, d_model, max_len=500):
        """
        Create a new PositionalEncoding.

        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
        """
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices, cosine to odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]


class STGNNTransformer(nn.Module):
    """
    Spatial-Temporal GNN-Transformer model for RUL prediction.

    The model has two main parts:
    1. GNN: captures relationships between sensors
    2. Transformer: captures patterns over time
    """

    def __init__(self, config, init_adj_matrix):
        """
        Create a new STGNNTransformer.

        Args:
            config: Dictionary with model configuration
            init_adj_matrix: Initial adjacency matrix for the GNN
        """
        super().__init__()

        self.num_nodes = config["num_nodes"]
        gnn_hidden = config["gnn_hidden_dim"]
        d_model = config["trans_d_model"]

        # Learnable adjacency matrix (starts from init_adj_matrix)
        self.adj_parameter = nn.Parameter(init_adj_matrix.float())

        # GNN layer
        self.gcn = GraphConvolutionLayer(
            input_dim=config["input_features"],
            output_dim=gnn_hidden
        )

        # Linear layer to prepare input for transformer
        self.flatten_dim = self.num_nodes * gnn_hidden
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

        # Final layer for RUL prediction
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def get_normalized_adj(self):
        """
        Normalize the adjacency matrix.

        Makes it symmetric and applies degree normalization.
        """
        # Make symmetric
        A = 0.5 * (self.adj_parameter + self.adj_parameter.t())

        # Add self-loops
        A = A + torch.eye(A.size(0), device=A.device)

        # Degree normalization
        deg = torch.sum(torch.abs(A), dim=1)
        D_inv_sqrt = torch.diag(torch.pow(deg + 1e-6, -0.5))

        return D_inv_sqrt @ A @ D_inv_sqrt

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch, time_steps, num_sensors, features)

        Returns:
            RUL prediction (batch, 1)
        """
        B, T, N, F = x.shape
        A = self.get_normalized_adj()

        # GNN: process each time step
        x = x.reshape(B * T, N, F)
        gcn_out = self.gcn(x, A)

        # Prepare sequence for transformer
        seq = gcn_out.reshape(B, T, -1)
        seq = self.embedding(seq)
        seq = self.pos_encoder(seq)

        # Transformer: process the sequence
        out = self.transformer_encoder(seq)

        # Average over time and predict RUL
        final_state = out.mean(dim=1)
        return self.regressor(final_state)
