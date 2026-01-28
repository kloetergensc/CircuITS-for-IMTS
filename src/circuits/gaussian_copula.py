import math

import torch
from torch import nn


class compute_correlation_matrix(nn.Module):
    """
    Gaussian Copula that models dependencies between N tokens.
    The correlation matrix is parametrized by token embeddings.
    """

    def __init__(self, embedding_dim, hidden_dim=16):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Network to predict correlation matrix from embeddings
        self.corr_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.hidden_dim = hidden_dim

    def forward(self, embeddings, mask=None):
        """
        Compute correlation matrix from token embeddings.

        Args:
            embeddings: (batch_size, N, embedding_dim)
            mask: (batch_size, N) - 1 for real tokens, 0 for padding
        Returns:
            corr_matrix: (batch_size, N, N) correlation matrix
        """
        orginial_emb_dim = len(embeddings.shape)
        if len(embeddings.shape) == 3:
            batch_size, N, _ = embeddings.shape
        if len(embeddings.shape) == 4:
            batch_size, K, N, _ = embeddings.shape
            embeddings = embeddings.flatten(end_dim=-3)
            mask = mask.flatten(end_dim=-2)

        # Concatenate pairs

        # Predict correlations for each pair
        embeddings_tanh = self.corr_net(embeddings)[:, :, : self.hidden_dim].clamp(min=-0.99, max=0.99)  # (B, N, D/2)
        # embeddings_sigmoid = self.corr_net(embeddings)[:, :, self.hidden_dim // 2 :]  # (B, N, D/2)
        corr = torch.matmul(embeddings_tanh, embeddings_tanh.transpose(1, 2)) / (self.hidden_dim)  # essentially this determines wether a correlation is positive or negative
        id_mask = torch.eye(N, device=corr.device, dtype=int).unsqueeze(0).expand_as(corr)

        # Ensure positive definiteness via spectral decomposition
        # corr = self.make_positive_definite(corr)
        if mask is not None:
            pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)  # (B, N, N)
            # Keep diagonal as 1, mask off-diagonal
            corr *= pair_mask
        corr = (corr * (1 - id_mask)) + id_mask
        assert (corr.abs() > 1).any, "corr > 1!"
        if orginial_emb_dim == 4:
            corr = corr.unsqueeze(1).reshape(batch_size, K, N, N)
        return corr


class compute_gaussian_parameters(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.corr_net = compute_correlation_matrix(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        self.std_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.mean_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def gaussian_marginals(self, embeddings, mask=None):
        means = self.mean_net(embeddings).squeeze(-1) * mask
        stds = torch.exp(self.std_net(embeddings).squeeze(-1)).clamp(min=1e-10)
        stds[~mask.to(bool)] = 1
        # Compute covariance matrix from stds and correlation matrix

        return means, stds

    def corr(self, embeddings, mask=None):
        return self.corr_net(embeddings, mask)
