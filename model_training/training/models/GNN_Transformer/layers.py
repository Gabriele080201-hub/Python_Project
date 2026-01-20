import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolutionLayer(nn.Module):
    """
    Graph Convolution Layer:
    H = GELU( LayerNorm( A @ X @ W + b ) )
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