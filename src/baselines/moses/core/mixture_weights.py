"""
Mixture Weights Computation Module

This module implements attention-based mixture weight computation for Gaussian mixture models.
It uses a learnable attention mechanism to compute weights that determine how much each
Gaussian component contributes to the final mixture.

The attention mechanism allows the model to dynamically focus on different parts of the
encoded history when computing mixture weights, making the model more flexible and expressive.
"""

import math
import pdb

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e6)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class multiheadattention(nn.Module):
    def __init__(self, n_hiddens: int, n_heads: int):
        super(multiheadattention, self).__init__()
        self.proj_param = nn.Linear(n_hiddens, n_hiddens * n_heads)
        self.linear_o = nn.Linear(n_heads * n_hiddens, n_hiddens)
        self.n_hiddens = n_hiddens
        self.n_heads = n_heads

    def forward(
        self,
        attn_queries: Tensor,
        attn_keys: Tensor,
        attn_vals: Tensor,
        attn_mask: Tensor,
    ) -> Tensor:
        r"""compute mha"""

        q = self.proj_param(attn_queries)
        k = self.proj_param(attn_keys)
        v = self.proj_param(attn_vals)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)

        mask = attn_mask.repeat(self.n_heads, 1, 1)

        x = ScaledDotProductAttention()(q, k, v, mask)
        x = self._reshape_from_batches(x)
        x = attn_queries + self.linear_o(x)
        return x

    def _reshape_to_batches(self, x: Tensor) -> Tensor:
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.n_heads
        return (
            x.reshape(batch_size, seq_len, self.n_heads, sub_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.n_heads, seq_len, sub_dim)
        )

    def _reshape_from_batches(self, x: Tensor) -> Tensor:
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.n_heads
        out_dim = in_feature * self.n_heads
        return (
            x.reshape(batch_size, self.n_heads, seq_len, in_feature)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, out_dim)
        )


class MixtureWeights(nn.Module):
    """
    Computes mixture weights for Gaussian mixture models using attention mechanism.

    This module uses learnable component embeddings (beta parameters) as queries to attend
    over the encoded history, producing mixture weights that sum to 1.0 via softmax.

    The attention mechanism allows each mixture component to focus on different aspects
    of the historical data, leading to more expressive and adaptive mixture models.

    Args:
        latent_dim: Dimension of the latent/hidden representations
        num_components: Number of mixture components (Gaussians)
        n_heads: Number of attention heads (default: 2)
    """

    def __init__(self, latent_dim: int, num_components: int, n_heads: int = 2) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.num_components = num_components
        self.n_heads = n_heads

        # Learnable component embeddings (queries for attention)
        self.mixture_wt_param = nn.Parameter(torch.randn(num_components, latent_dim))

        # Multi-head attention for mixture weights
        self.mha_wts = multiheadattention(latent_dim, n_heads)

        # Output projection to scalar weights
        self.mix_wts_nn = nn.Linear(latent_dim, 1)

    def forward(self, encoded_history: Tensor, sequence_mask: Tensor) -> Tensor:
        """
        Compute mixture weights using attention over encoded history.

        The process:
        1. Use component embeddings as queries to attend over encoded history
        2. Compute multi-head attention weights
        3. Project to scalar weights and apply softmax for normalization

        Args:
            encoded_history: Encoded historical data [batch_size, seq_len, latent_dim]
                            This corresponds to 'x' in the second implementation
            sequence_mask: Binary mask for valid observations [batch_size, seq_len]
                          (1 for valid positions, 0 for padded/invalid positions)

        Returns:
            Mixture weights: [batch_size, num_components]
                           Each row sums to 1.0 (valid probability distribution)
        """
        batch_size = encoded_history.shape[0]

        # Create mask for mixture weight attention
        wts_mask = torch.ones_like(self.mixture_wt_param[:, 0:1])
        attn_mix_wts_mask = torch.matmul(
            wts_mask[None, :, :].repeat(batch_size, 1, 1), sequence_mask.unsqueeze(-2)
        )

        # Expand mixture weight parameters for batch
        wts_query = self.mixture_wt_param[None, :, :].repeat(batch_size, 1, 1)

        # Apply multi-head attention
        mw_ = self.mha_wts(
            wts_query, encoded_history, encoded_history, attn_mix_wts_mask
        )

        # Project to scalar weights
        mw = self.mix_wts_nn(mw_).squeeze(-1)

        # Apply softmax to ensure weights sum to 1.0
        mw = nn.LogSoftmax(-1)(mw)

        return mw
