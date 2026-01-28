"""MOSES: Mixture of Splines for Enhanced Sampling.

This module implements MOSES, a probabilistic model that combines:
- Transformer-based encoding for time series conditioning
- Mixture of multivariate Gaussians as base distribution
- Rational linear spline flows for flexible transformations
- Mixture weighting for multi-component modeling

Example:
    >>> model = Moses(n_inputs=5, latent_dim=128, num_components=3)
    >>> model.to('cuda')
    >>> # Forward pass
    >>> nll = model.compute_loss(tx, cx, mx, x, tq, cq, mq, y)
    >>> # Sampling
    >>> samples = model.sample(mq, num_samples=1000)
"""

__all__ = [
    "Moses",
]

import pdb
import warnings
from typing import Final, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import ModuleList

from .distributions.base_distribution import MultiGaussian
from .distributions.spline_transforms import RationalLinearSplineFlow
from .encoders.grafiti.grafiti import GraFITi
from .encoders.transformer import TimeSeriesEncoder as transformer
from .mixture_weights import MixtureWeights
from .utils.metrics import compute_mnll_on_latent, compute_njnll_on_latent


class Moses(nn.Module):
    """MOSES: Mixture of Splines for Enhanced Sampling.

    A probabilistic model combining transformer encoding, mixture of Gaussians,
    and normalizing flows for flexible density modeling and sampling.

    Args:
        n_inputs: Number of input features
        n_heads: Number of attention heads
        num_components: Number of mixture components
        latent_dim: Dimensionality of latent space
        num_flow_layers: Number of flow transformation layers
        num_bins: Number of bins for spline flows
        bounds: Bounds for spline transformations
        device: Device for model parameters
    """

    def __init__(
        self,
        n_inputs: int = 5,
        n_heads: int = 2,
        num_components: int = 2,
        latent_dim: int = 128,
        num_flow_layers: int = 3,
        num_encoder_layers: int = 1,
        num_bins: int = 16,
        bounds: float = 20.0,
        encoder_model: str = "transformer",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        # Store configuration
        self.config = {
            "n_inputs": n_inputs,
            "n_heads": n_heads,
            "num_components": num_components,
            "latent_dim": latent_dim,
            "num_flow_layers": num_flow_layers,
            "num_encoder_layers": num_encoder_layers,
            "num_bins": num_bins,
            "encoder_model": encoder_model,
            "bounds": bounds,
        }

        # Initialize components
        self._build_model()

        # State variables (computed during forward pass)
        self._reset_state()

        # Move to device if specified
        if device is not None:
            self.to(device)

    def _build_model(self) -> None:
        """Build all model components."""
        cfg = self.config

        # Normalizing flows
        self.flows = ModuleList(
            [
                RationalLinearSplineFlow(
                    d_model=cfg["latent_dim"],
                    tail_bound=cfg["bounds"],
                    num_bins=cfg["num_bins"],
                )
                for _ in range(cfg["num_flow_layers"])
            ]
        )

        # Base distribution
        self.base_distribution = MultiGaussian(latent_dim=cfg["latent_dim"])

        # Mixture weights
        self.mixture_weights = MixtureWeights(
            latent_dim=cfg["latent_dim"], num_components=cfg["num_components"]
        )
        if cfg["encoder_model"] == "grafiti":
            self.encoder = GraFITi(
                input_dim=cfg["n_inputs"],
                latent_dim=cfg["latent_dim"],
                n_gaussians=cfg["num_components"],
                attn_head=cfg["n_heads"],
                n_layers=cfg["num_encoder_layers"],
            )
        else:
            self.encoder = transformer(
                n_channels=cfg["n_inputs"],
                d_model=cfg["latent_dim"],
                n_gaussians=cfg["num_components"],
                n_heads=cfg["n_heads"],
                num_layers=cfg["num_encoder_layers"],
            )
        # Encoder

    def _reset_state(self) -> None:
        """Reset internal state variables."""
        self.log_mixture_weights: Optional[Tensor] = None

    @property
    def num_components(self) -> int:
        """Number of mixture components."""
        return self.config["num_components"]

    @property
    def latent_dim(self) -> int:
        """Latent space dimensionality."""
        return self.config["latent_dim"]

    def forward(
        self,
        tx: Tensor,
        cx: Tensor,
        mx: Tensor,
        x: Tensor,
        tq: Tensor,
        cq: Tensor,
        mq: Tensor,
    ):
        """Encode conditioning information.

        Args:
            tx, cx, mx, x: Training/context inputs
            tq, cq, mq: Query inputs

        Returns:
            Tuple of (full_conditioning, history_conditioning, log_mixture_weights)
        """
        # Encode inputs
        history_encoding, query_encoding = self.encoder(tx, cx, mx, x, tq, cq, mq)

        # Compute mixture weights
        self.log_mixture_weights = self.mixture_weights(history_encoding, mx)
        # Update base distribution
        self.base_distribution(query_encoding, mq)

        # Update flow conditioning
        for flow in self.flows:
            flow(query_encoding)

    def _apply_flows_forward(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply normalizing flows in forward direction.

        Args:
            y: Input tensor

        Returns:
            Tuple of (transformed_y, log_det_jacobian)
        """
        ldj = torch.zeros_like(y)

        for flow in self.flows:
            y, ldj_step = flow.forward_(y)
            ldj = ldj + ldj_step

        return y, ldj

    def _apply_flows_inverse(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply normalizing flows in inverse direction.

        Args:
            z: Latent tensor

        Returns:
            Tuple of (transformed_z, log_det_jacobian)
        """
        ldj = torch.zeros_like(z)

        for flow in reversed(self.flows):
            z, ldj_step = flow.inverse_(z)
            ldj = ldj + ldj_step

        return z, ldj

    def _prepare_target_tensor(self, y: Tensor) -> Tensor:
        """Prepare target tensor for mixture processing.

        Args:
            y: Target tensor of shape (batch_size, target_dim)

        Returns:
            Expanded tensor of shape (batch_size, num_components, target_dim)
        """
        return y.unsqueeze(1).repeat(1, self.num_components, 1)

    def compute_njnll(self, y: Tensor, mq: Tensor) -> Tensor:
        """Compute normalized joint negative log-likelihood.

        Args:
            y: Target values
            mq: Query mask

        Returns:
            Mean NJNLL across batch
        """
        if self.log_mixture_weights is None:
            raise RuntimeError(
                "Must call encode_conditioning() before computing likelihood"
            )
        mq_expanded = mq[:, None, :]
        y_expanded = self._prepare_target_tensor(y)
        z, ldj = self._apply_flows_forward(y_expanded)
        z = z * mq_expanded
        ldj = ldj * mq_expanded
        likelihood = compute_njnll_on_latent(
            z,
            self.base_distribution,
            ldj,
            mq,
            self.log_mixture_weights,
        )

        return likelihood.mean()

    def compute_mnll(self, y: Tensor, mq: Tensor) -> Tensor:
        """Compute marginal negative log-likelihood.

        Args:
            y: Target values
            mq: Query mask

        Returns:
            Mean MNLL across batch
        """
        if self.log_mixture_weights is None:
            raise RuntimeError(
                "Must call encode_conditioning() before computing likelihood"
            )

        y_expanded = self._prepare_target_tensor(y)
        z, ldj = self._apply_flows_forward(y_expanded)

        likelihood = compute_mnll_on_latent(
            z,
            self.base_distribution,
            ldj,
            mq,
            self.log_mixture_weights,
        )

        return likelihood.mean()

    def _sample_mixture_indices(self, num_samples: Optional[int] = None) -> Tensor:
        """Sample indices from mixture weights.

        Args:
            batch_size: Optional batch size override

        Returns:
            Sampled indices
        """
        if self.log_mixture_weights is None:
            raise RuntimeError("Must call encode_conditioning() before sampling")
        mixture_probs = torch.exp(self.log_mixture_weights)

        if num_samples is None:
            num_samples = 1

        return torch.multinomial(mixture_probs, num_samples, replacement=True)

    def sample_joint(
        self, mq: Tensor, num_samples: int = 1000, return_indices: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Sample from joint distribution.

        Args:
            mq: Query mask
            num_samples: Number of samples to generate
            return_indices: Whether to return mixture indices

        Returns:
            Samples tensor, optionally with mixture indices
        """
        if self.base_distribution.mean is None:
            raise RuntimeError("Must call encode_conditioning() before sampling")
            # Sample from base distribution
        z = self.base_distribution.sample_joint(num_samples)
        z *= mq[:, None, :, None]
        # Transform through flows

        x, _ = self._apply_flows_inverse(z)  # B, D, K, nsamples

        # Sample mixture indices
        indices = self._sample_mixture_indices(num_samples=num_samples)  # B, nsamples

        # Select according to mixture indices

        indices_exp = (
            indices.unsqueeze(-1).expand(-1, -1, mq.shape[1]).permute(0, 2, 1)
        )  # (B, K, nsamples)
        # For gather, X must be permuted so that D is at the last dimension
        x_perm = x.permute(0, 2, 3, 1)  # (B, K, nsamples, D)

        # Gather along last dimension
        x_selected = torch.gather(
            x_perm, dim=-1, index=indices_exp.unsqueeze(-1)
        ).squeeze(
            -1
        )  # (B, K, nsamples)

        x_selected = x_selected * mq.unsqueeze(-1)
        return (x_selected, indices) if return_indices else x_selected

    def sample_marginal(
        self, mq: Tensor, num_samples: int = 1000, return_indices: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Sample from marginal distributions.

        Args:
            mq: Query mask
            num_samples: Number of samples to generate
            return_indices: Whether to return mixture indices

        Returns:
            Samples tensor, optionally with mixture indices
        """
        if self.base_distribution.mean is None:
            raise RuntimeError("Must call encode_conditioning() before sampling")

        # Sample from marginal base distribution
        z = self.base_distribution.sample_marginal(num_samples)
        z *= mq[:, None, :, None]
        # Transform through flows
        x, _ = self._apply_flows_inverse(z)
        seq_len = mq.shape[-1]
        # Sample mixture indices
        indices = self._sample_mixture_indices(num_samples * seq_len)

        # Select according to mixture indices
        batch_indices = torch.arange(x.shape[0]).unsqueeze(1)
        x_selected = x[batch_indices, indices]
        x_selected = x_selected * mq.unsqueeze(-1)
        return (x_selected, indices) if return_indices else x_selected

    def sample(
        self,
        mq: Tensor,
        num_samples: int = 1000,
        mode: str = "joint",
        return_indices: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Unified sampling interface.

        Args:
            mq: Query mask
            num_samples: Number of samples to generate
            mode: Sampling mode ('joint' or 'marginal')
            return_indices: Whether to return mixture indices

        Returns:
            Samples tensor, optionally with mixture indices
        """
        if mode == "joint":
            return self.sample_joint(mq, num_samples, return_indices)
        elif mode == "marginal":
            return self.sample_marginal(mq, num_samples, return_indices)
        else:
            raise ValueError(
                f"Invalid sampling mode: {mode}. Use 'joint' or 'marginal'"
            )

    def get_config(self) -> dict:
        """Get model configuration."""
        return self.config.copy()

    def extra_repr(self) -> str:
        """String representation of model configuration."""
        return (
            f"latent_dim={self.config['latent_dim']}, "
            f"num_components={self.config['num_components']}, "
            f"num_flow_layers={self.config['num_flow_layers']}"
        )
