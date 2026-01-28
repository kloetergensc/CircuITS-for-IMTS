"""
Transformer Architecture for Time Series Encoding

This module implements a Transformer-based encoder specifically designed for time series data
with multiple channels. It uses multi-head attention mechanisms to encode observations and
queries, supporting masked attention and Gaussian mixture outputs.

Key Features:
- Scaled dot-product attention with optional masking
- Multi-head attention for parallel processing
- Time and channel embeddings for time series data
- Cross-attention between queries and observations
- Gaussian mixture model support
"""

import math
import pdb
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ScaledDotProductAttention(nn.Module):
    """
    Implements scaled dot-product attention mechanism.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Args:
        query: Query tensor of shape (..., seq_len, d_k)
        key: Key tensor of shape (..., seq_len, d_k)
        value: Value tensor of shape (..., seq_len, d_v)
        mask: Optional attention mask (0 for positions to ignore)

    Returns:
        Attention-weighted values of shape (..., seq_len, d_v)
    """

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        # Calculate attention scores
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask if provided (set masked positions to large negative value)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e6)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        return torch.matmul(attention_weights, value)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Allows the model to jointly attend to information from different representation
    subspaces at different positions.

    Args:
        d_model: Model dimension (hidden size)
        n_heads: Number of attention heads
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Linear projections for queries, keys, and values
        self.w_query = nn.Linear(d_model, d_model * n_heads, bias=False)
        self.w_key = nn.Linear(d_model, d_model * n_heads, bias=False)
        self.w_value = nn.Linear(d_model, d_model * n_heads, bias=False)

        # Output projection
        self.w_output = nn.Linear(n_heads * d_model, d_model)

        # Attention mechanism
        self.attention = ScaledDotProductAttention()

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass for multi-head attention.

        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Output tensor with residual connection [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = query.size(0), query.size(1)

        # Apply linear transformations and split into heads
        Q = self._split_heads(self.w_query(query), batch_size)
        K = self._split_heads(self.w_query(key), batch_size)
        V = self._split_heads(self.w_query(value), batch_size)

        # Expand mask for multiple heads if provided
        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)

        # Apply attention
        attention_output = self.attention(Q, K, V, mask)

        # Concatenate heads and apply output projection
        concatenated = self._concatenate_heads(attention_output, batch_size)
        output = self.w_output(concatenated)

        # Residual connection
        return query + output

    def _split_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """Split the last dimension into multiple attention heads."""
        seq_len = x.size(1)
        return (
            x.view(batch_size, seq_len, self.n_heads, self.d_model)
            .transpose(1, 2)  # [batch_size, n_heads, seq_len, d_model]
            .contiguous()
            .view(batch_size * self.n_heads, seq_len, self.d_model)
        )

    def _concatenate_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """Concatenate attention heads back together."""
        seq_len, d_model = x.size(1), x.size(2)
        return (
            x.view(batch_size, self.n_heads, seq_len, d_model)
            .transpose(1, 2)  # [batch_size, seq_len, n_heads, d_model]
            .contiguous()
            .view(batch_size, seq_len, self.n_heads * d_model)
        )


class TimeSeriesEncoder(nn.Module):
    """
    Transformer encoder for time series data with multiple channels.

    This encoder processes time series observations and queries, applying:
    1. Time and channel embeddings
    2. Self-attention on observations
    3. Cross-attention between queries and observations
    4. Support for Gaussian mixture model outputs

    Args:
        n_channels: Number of input channels
        d_model: Model dimension (hidden size)
        n_gaussians: Number of Gaussian components for mixture model
        n_heads: Number of attention heads (default: 2)
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int,
        n_gaussians: int,
        n_heads: int = 2,
        num_layers: int = 1,
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.d_model = d_model
        self.n_gaussians = n_gaussians
        self.n_heads = n_heads

        # Time embeddings (using sinusoidal encoding)
        self._init_time_embeddings()

        # Channel embeddings
        self._init_channel_embeddings()

        # Query and key projections
        self.query_projection = nn.Linear(2 * d_model, d_model)
        self.key_projection = nn.Linear(2 * d_model + 1, d_model)

        # Multi-head attention layers
        self.self_attention = nn.ModuleList()
        for _ in range(num_layers):
            self.self_attention.append(MultiHeadAttention(d_model, n_heads))
        self.cross_attention = MultiHeadAttention(d_model, n_heads)

        # Output projection for Gaussian mixture
        self.gaussian_projection = nn.Linear(d_model, d_model * n_gaussians)

        # Activation function
        self.activation = nn.ReLU()

    def _init_time_embeddings(self) -> None:
        """Initialize time embedding layers."""
        # Identity and sinusoidal time embeddings
        self.time_embed_query_0 = nn.Linear(1, 1)
        self.time_embed_query_sin = nn.Linear(1, self.d_model - 1)

        self.time_embed_obs_0 = nn.Linear(1, 1)
        self.time_embed_obs_sin = nn.Linear(1, self.d_model - 1)

    def _init_channel_embeddings(self) -> None:
        """Initialize channel embedding layers."""
        self.channel_embed_obs = nn.Linear(self.n_channels, self.d_model)
        self.channel_embed_query = nn.Linear(self.n_channels, self.d_model)

    def _encode_time(self, timestamps: Tensor, is_query: bool = True) -> Tensor:
        """
        Encode timestamps using identity + sinusoidal embeddings.

        Args:
            timestamps: Time values [batch_size, seq_len, 1]
            is_query: Whether encoding query or observation timestamps

        Returns:
            Time embeddings [batch_size, seq_len, d_model]
        """
        if is_query:
            time_0 = self.time_embed_query_0(timestamps)
            time_sin = torch.sin(self.time_embed_query_sin(timestamps))
        else:
            time_0 = self.time_embed_obs_0(timestamps)
            time_sin = torch.sin(self.time_embed_obs_sin(timestamps))

        return torch.cat([time_0, time_sin], dim=-1)

    def _encode_channels(self, channels: Tensor, is_query: bool = True) -> Tensor:
        """
        Encode channel indices using one-hot encoding + linear projection.

        Args:
            channels: Channel indices [batch_size, seq_len]
            is_query: Whether encoding query or observation channels

        Returns:
            Channel embeddings [batch_size, seq_len, d_model]
        """
        # Convert to one-hot encoding
        channel_onehot = F.one_hot(
            channels.to(torch.int64), num_classes=self.n_channels
        ).to(channels.dtype)

        # Apply embedding layer
        if is_query:
            channel_embed = self.activation(self.channel_embed_query(channel_onehot))
        else:
            channel_embed = self.activation(self.channel_embed_obs(channel_onehot))

        return channel_embed

    def _create_attention_masks(
        self, obs_mask: Tensor, query_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Create attention masks for self-attention and cross-attention.

        Args:
            obs_mask: Observation mask [batch_size, n_obs]
            query_mask: Query mask [batch_size, n_queries]

        Returns:
            Tuple of (self_attention_mask, cross_attention_mask)
        """
        # Self-attention mask (obs to obs)
        self_mask = torch.matmul(obs_mask.unsqueeze(-1), obs_mask.unsqueeze(-2))

        # Cross-attention mask (query to obs)
        cross_mask = torch.matmul(query_mask.unsqueeze(-1), obs_mask.unsqueeze(-2))

        return self_mask, cross_mask

    def forward(
        self,
        obs_times: Tensor,
        obs_channels: Tensor,
        obs_mask: Tensor,
        obs_values: Tensor,
        query_times: Tensor,
        query_channels: Tensor,
        query_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the time series encoder.

        Args:
            observations: Observation data [batch_size, n_obs, 3]
                         (time, channel, value)
            obs_mask: Observation mask [batch_size, n_obs]
            queries: Query data [batch_size, n_queries, 2]
                    (time, channel)
            query_mask: Query mask [batch_size, n_queries]

        Returns:
            Tuple of:
                - History encoding [batch_size, n_obs, d_model]
                - Full encoding [batch_size, n_gaussians, n_obs, d_model]
        """

        num_queries = query_times.shape[-1]
        # Encode time and channel information
        obs_time_embed = self._encode_time(obs_times.unsqueeze(-1), is_query=False)
        query_time_embed = self._encode_time(query_times.unsqueeze(-1), is_query=True)

        obs_channel_embed = self._encode_channels(obs_channels, is_query=False)
        query_channel_embed = self._encode_channels(query_channels, is_query=True)

        # Combine embeddings
        obs_combined = torch.cat(
            [obs_time_embed, obs_channel_embed, obs_values.unsqueeze(-1)], dim=-1
        )
        query_combined = torch.cat([query_time_embed, query_channel_embed], dim=-1)

        # Apply masks to embeddings
        obs_combined = obs_combined * obs_mask.unsqueeze(-1)
        query_combined = query_combined * query_mask.unsqueeze(-1)

        # Project to model dimension
        obs_projected = self.key_projection(obs_combined)
        query_projected = self.query_projection(query_combined)

        # Create attention masks
        self_mask, cross_mask = self._create_attention_masks(obs_mask, query_mask)

        # Apply self-attention to observations

        for layer in self.self_attention:
            obs_projected = layer(
                obs_projected, obs_projected, obs_projected, self_mask
            )
        history_encoding = self.activation(obs_projected)

        # Apply cross-attention between queries and observations
        query_attended = self.cross_attention(
            query_projected, history_encoding, history_encoding, cross_mask
        )
        query_encoding = self.activation(query_attended)
        # Project for Gaussian mixture model
        query_encoding_full = self.gaussian_projection(query_encoding)
        query_encoding = query_encoding_full.reshape(
            -1, num_queries, self.n_gaussians, self.d_model
        )

        # Transpose for output format [batch_size, n_gaussians, seq_len, d_model]
        query_encoding = query_encoding.permute(0, 2, 1, 3)
        return history_encoding, query_encoding
