"""Mixture of Gaussians with attention-based covariance computation.

This module provides a PyTorch implementation of a mixture of multivariate Gaussians
that uses an attention-style mechanism to compute covariance matrices. The implementation
supports both marginal and joint sampling from the learned distributions.

Example:
    >>> model = MultiGaussian(latent_dim=128)
    >>> x = torch.randn(2, 3, 10, 128)  # (batch, components, queries, features)
    >>> mask = torch.ones(2, 3, 10)     # (batch, components, queries)
    >>> mean, cov = model(x, mask)
    >>> joint_samples = model.sample_joint(num_samples=100)
    >>> marginal_samples = model.sample_marginal(num_samples=1000)
"""

import math
import pdb
from typing import Optional, Tuple

import torch
from networkx import number_connected_components
from torch import Tensor, nn
from torch.distributions import MultivariateNormal


class MultiGaussian(nn.Module):
    """Compute parameters and sample from a mixture of multivariate Gaussians.

    This class uses an attention-style mechanism to compute covariance matrices
    and provides methods for both marginal and joint sampling.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.mean_layer = nn.Linear(latent_dim, 1)
        self.query_layer = nn.Linear(latent_dim, latent_dim)

        # Cache for computed parameters
        self._mean: Optional[Tensor] = None
        self._cov: Optional[Tensor] = None
        self._U: Optional[Tensor] = (
            None  # cov = U@U^T, useful for Woodbury identiy if K>M
        )

    @property
    def mean(self) -> Optional[Tensor]:
        """Cached mean parameters."""
        return self._mean

    @property
    def cov(self) -> Optional[Tensor]:
        """Cached covariance parameters."""
        return self._cov

    @property
    def U(self) -> Optional[Tensor]:
        """Cached covariance parameters."""
        return self._U

    def _compute_attention_scores(self, queries: Tensor) -> Tensor:
        """Compute scaled dot-product attention scores."""
        return torch.matmul(queries, queries.transpose(-2, -1)) / math.sqrt(
            self.latent_dim
        )

    def _create_masked_covariance(self, scores: Tensor, mask: Tensor) -> Tensor:
        """Create covariance matrix with masking and identity regularization."""
        b, k = mask.shape

        c = scores.shape[1]
        mask = mask.unsqueeze(1).repeat(1, c, 1)

        # Create identity matrix
        identity = torch.eye(k, device=scores.device, dtype=scores.dtype)
        identity = identity.expand(b, c, -1, -1)

        # Create attention mask: (b, c, k) -> (b, c, k, k)
        attn_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        # Apply masking: use attention scores where both positions are valid,
        # otherwise use identity
        return torch.where(attn_mask, scores + identity, identity)

    def compute_covariance(self, h: Tensor, mask: Tensor) -> Tensor:
        """Compute covariance matrix using attention mechanism.

        Args:
            h: Input tensor (batch_size, num_components, num_queries, hidden_dim)
            mask: Binary mask (batch_size, num_queries)

        Returns:
            Covariance tensor (batch_size, num_components, num_queries, num_queries)
        """
        queries = self.query_layer(h)
        self._U = queries * mask.unsqueeze(1).unsqueeze(-1)
        scores = self._compute_attention_scores(queries)
        return self._create_masked_covariance(scores, mask)

    def forward(self, h: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute and cache mean and covariance parameters.

        Args:
            x: Input tensor (batch_size, num_components, num_queries, hidden_dim)
            mask: Binary mask (batch_size, num_components, num_queries)

        Returns:
            Tuple of (mean, covariance) tensors
        """
        self.reset_cache()
        self._mean = self.mean_layer(h).squeeze(-1) * mask[:, None, :]
        self._cov = self.compute_covariance(h, mask.bool())
        # Ensure diagonal elements are at least 1.0
        # batch_size, num_components, num_queries, _ = self._cov.shape
        # eye = torch.eye(num_queries, device=self._cov.device, dtype=self._cov.dtype)
        # eye = eye.expand(batch_size, num_components, -1, -1)
        # self._cov = self._cov + eye * (1.0 - torch.diagonal(self._cov, dim1=-2, dim2=-1).unsqueeze(-1))
        if self._mean is None or self._cov is None:
            raise RuntimeError("Mean and covariance must not be None")

    def sample_marginal(self, num_samples: int = 1000) -> Tensor:
        """Generate samples from marginal distributions (diagonal covariance only).

        Args:
            num_samples: Number of samples to generate

        Returns:
            Samples tensor (*mean.shape, num_samples)
        """
        if self._mean is None or self._cov is None:
            raise RuntimeError("Must call forward() before sampling")

        # Extract diagonal standard deviations
        std = torch.diagonal(self._cov, dim1=-2, dim2=-1).sqrt()

        # Generate samples: mean + std * noise
        noise = torch.randn(*self._mean.shape, num_samples, device=self._mean.device)
        return self._mean.unsqueeze(-1) + std.unsqueeze(-1) * noise

    def sample_joint(self, num_samples: int = 1000) -> Tensor:
        """Generate samples from joint multivariate normal distributions.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Samples tensor (batch_size, num_components, num_queries, num_samples)
        """
        if self._mean is None or self._cov is None:
            raise RuntimeError("Must call forward() before sampling")

        batch_size, num_components, num_queries = self._mean.shape

        # Reshape for batch processing
        mean_flat = self._mean.view(-1, num_queries)
        cov_flat = self._cov.view(-1, num_queries, num_queries)

        # Create and sample from multivariate normal
        try:
            mvn = MultivariateNormal(mean_flat, covariance_matrix=cov_flat)
            samples_flat = mvn.sample(
                torch.Size(
                    [
                        num_samples,
                    ]
                )
            )  # (num_samples, batch*components, queries)
        except RuntimeError as e:
            # Handle potential numerical issues with covariance matrix
            raise RuntimeError(
                f"Failed to create multivariate normal distribution: {e}"
            ) from e

        # Reshape back to original structure
        samples = samples_flat.permute(1, 2, 0).view(
            batch_size, num_components, num_queries, num_samples
        )

        return samples

    def reset_cache(self) -> None:
        """Clear cached parameters."""
        self._mean = None
        self._cov = None
        self._cov = None
