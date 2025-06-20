"""PyTorch implementation of the custom DEC clustering layer."""

import torch
from torch import nn


class DECLayer(nn.Module):
    """Student t-distribution based soft assignment layer used in DEC."""

    def __init__(self, n_clusters, input_dim, weights=None, alpha=1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.empty(n_clusters, input_dim))
        nn.init.xavier_uniform_(self.clusters)
        if weights is not None:
            with torch.no_grad():
                self.clusters.copy_(torch.tensor(weights, dtype=torch.float32))

    def forward(self, inputs):
        q = 1.0 / (
            1.0
            + torch.sum((inputs.unsqueeze(1) - self.clusters) ** 2, dim=2)
            / self.alpha
        )
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q
