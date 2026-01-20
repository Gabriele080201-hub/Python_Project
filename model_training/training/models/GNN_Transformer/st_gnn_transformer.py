import torch
import torch.nn as nn

from .layers import GraphConvolutionLayer
from .positional_encoding import PositionalEncoding


class STGNNTransformer(nn.Module):
    """
    Spatial-Temporal GNN + Transformer for RUL prediction.
    """

    def __init__(self, config, init_adj_matrix):
        super().__init__()

        self.num_nodes = config["num_nodes"]
        H_g = config["gnn_hidden_dim"]
        d_model = config["trans_d_model"]

        self.adj_parameter = nn.Parameter(init_adj_matrix.float())

        self.gcn = GraphConvolutionLayer(
            input_dim=config["input_features"],
            output_dim=H_g
        )

        self.flatten_dim = self.num_nodes * H_g
        self.embedding = nn.Linear(self.flatten_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

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

        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def get_normalized_adj(self):
        A = 0.5 * (self.adj_parameter + self.adj_parameter.t())
        A = A + torch.eye(A.size(0), device=A.device)

        deg = torch.sum(torch.abs(A), dim=1)
        D_inv_sqrt = torch.diag(torch.pow(deg + 1e-6, -0.5))

        return D_inv_sqrt @ A @ D_inv_sqrt

    def forward(self, x):
        B, T, N, F = x.shape
        A = self.get_normalized_adj()

        x = x.reshape(B * T, N, F)
        gcn_out = self.gcn(x, A)

        seq = gcn_out.reshape(B, T, -1)
        seq = self.embedding(seq)
        seq = self.pos_encoder(seq)

        out = self.transformer_encoder(seq)
        final_state = out.mean(dim = 1)
        

        return self.regressor(final_state)