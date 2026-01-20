"""
Graph Neural Network Layer Module

This module contains the Graph Convolution Layer used in the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolutionLayer(nn.Module):
    """
    Graph Convolution Layer for processing sensor data as a graph.

    This layer learns relationships between different sensors by treating them
    as nodes in a graph. It applies the transformation:
    H = GELU(LayerNorm(A @ X @ W + b))

    where:
    - A is the adjacency matrix (sensor relationships)
    - X is the input features
    - W is learnable weights
    - b is learnable bias

    Args:
        input_dim (int): Number of input features per node
        output_dim (int): Number of output features per node
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        # Learnable weight matrix for feature transformation
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))

        # Learnable bias term
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))

        # Initialize weights with Xavier normalization (good for deep networks)
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

        # Layer normalization for stable training
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, adj):
        """
        Forward pass through the graph convolution layer.

        Args:
            x (torch.Tensor): Input features, shape (batch, num_nodes, input_dim)
            adj (torch.Tensor): Adjacency matrix, shape (num_nodes, num_nodes)

        Returns:
            torch.Tensor: Output features, shape (batch, num_nodes, output_dim)
        """
        # Step 1: Aggregate information from neighboring nodes using adjacency matrix
        support = torch.matmul(adj, x)

        # Step 2: Apply linear transformation (weights and bias)
        out = torch.matmul(support, self.weight) + self.bias

        # Step 3: Normalize output
        out = self.norm(out)

        # Step 4: Apply GELU activation function (smooth non-linearity)
        return F.gelu(out)
