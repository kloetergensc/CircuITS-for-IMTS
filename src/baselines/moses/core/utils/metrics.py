import math
import pdb
from typing import Any, Dict, Optional, Tuple

import numpy as np
import properscoring as ps
import torch
from scipy import stats
from scipy.spatial import cKDTree as KDTree
from torch import Tensor, nn


def compute_energy_score(yhat, y, mask, beta=1):
    """
    y: [batch, dim]
    yhat: [batch, dim, nsamples]
    mask: optional, [batch, dim]
    """

    nsamples = yhat.shape[-1]

    yhat = yhat.permute(0, 2, 1)

    valid_batch = mask.sum(dim=1) > 0
    y = y[valid_batch] * mask[valid_batch]
    yhat = yhat[valid_batch] * mask[valid_batch, None, :]

    # First term: mean over samples
    diff = torch.cdist(yhat, y[:, None, :], p=2)  # [batch, nsamples, 1]
    first_term = diff.pow(beta).mean(dim=1).squeeze(-1)  # mean over nsamples

    # Second term: pairwise distances, exclude diagonal
    pairwise = torch.cdist(yhat, yhat, p=2)
    diag = torch.diagonal(pairwise, dim1=1, dim2=2)
    pairwise = pairwise - torch.diag_embed(diag)
    second_term = -pairwise.pow(beta).sum(dim=(1, 2)) / (2 * nsamples**2)

    energy = first_term + second_term
    return energy.sum() / y.shape[0]  # per batch


def compute_mse(yhat, y, mask):
    squared_error = (y - yhat) ** 2
    sse = squared_error * mask
    mse = sse.sum() / mask.sum()
    return mse


def compute_crps(yhat, y, mask):
    y_expanded = y.unsqueeze(-1)  # [B, 1, K]

    term1 = torch.mean(torch.abs(y_expanded - yhat), dim=-1)  # [B, K]

    # Compute pairwise differences between ensemble members
    yhat1 = yhat.unsqueeze(-2)  # [B, K, 1, nsamples]
    yhat2 = yhat.unsqueeze(-1)  # [B, K, nsamples, 1]
    term2 = 0.5 * torch.mean(torch.abs(yhat1 - yhat2), dim=(-1, -2))  # [B, K]

    crps = term1 - term2
    crps *= mask
    # Apply mask
    return crps.sum() / mask.sum()


def multivariate_normal(z, mu, sigma, mq):
    r"""
    Log-density of a multivariate Gaussian using Cholesky decomposition.

    Formula:
        log p(x) = - (K / 2) * log(2π)
                - 0.5 * || L^{-1} (x - μ) ||^2
                - sum_i log L_ii (sum of diag?)

    Where:
        - x : data vector
        - μ : mean vector
        - L : lower-triangular Cholesky factor of covariance Σ (Σ = L @ L.T)
        - K : dimension of x

    """
    L = torch.linalg.cholesky(sigma)  # [..., D, D]
    diff = z - mu  # [..., D]
    solve_l_diff = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False).squeeze(-1)
    quad = (solve_l_diff**2).sum(-1)
    ldj = torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    k = mq.sum(-1)
    log_prob = -0.5 * (k[:, None] * math.log(2 * math.pi) + quad) - ldj
    return -log_prob


def multivariate_normal_woodbury(z, mu, U, mq):
    r"""
    Log-density of a multivariate Gaussian using Woodbury Identity.

    Formula:
        log p(x) = - (K / 2) * log(2π)
                - 0.5 * || L^{-1} (x - μ) ||^2
                - sum_i log L_ii (sum of diag?)

    Where:
        - x : data vector
        - μ : mean vector
        - U : Σ = I + U@U.T
        - K : dimension of x (number of queries)

    """
    U = U.clamp(min=-50, max=50)
    diff = (z - mu) * mq.unsqueeze(1)
    identity = torch.eye(U.shape[-1], device=U.device).unsqueeze(0).unsqueeze(0)
    # A = identity + (U.transpose(-1, -2) @ U)
    U_double = U.double()
    A = torch.eye(U.shape[-1], device=U.device, dtype=torch.float64) + (U_double.transpose(-1, -2) @ U_double)
    # print(torch.abs(U).max())
    L = torch.linalg.cholesky(A)
    ldj = torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    A_inv = torch.cholesky_inverse(L)
    Q = (U * diff.unsqueeze(-1)).sum(-2)
    # Quadratic term: sum_i (||y_i||^2 - (U^T y_i)^T A^-1 (U^T y_i))
    quad = torch.sum(diff**2, dim=-1) - (torch.sum((Q.unsqueeze(-1) * A_inv), dim=-2) * Q).sum(dim=-1)
    k = mq.sum(-1)
    log_prob = -0.5 * (k[:, None] * math.log(2 * math.pi) + quad) - ldj
    return -log_prob


def WassersteinMetric(model, mq, nsamples=1000):
    marginal_y = model.marginal_samples(nsamples=nsamples)  # samples drawn from the marginals
    marginal_y_numerical = model.joint_samples(nsamples=nsamples)  # samples drawn from the joints
    marginal_y_sorted = marginal_y.sort(-1)[0]  # sort the samples
    marginal_y_numerical_sorted = marginal_y_numerical.sort(-1)[0]
    dist = (marginal_y_sorted - marginal_y_numerical_sorted) ** 2
    dist_mean = (dist.mean(-1)) ** 0.5
    return dist_mean * mq


def compute_njnll_on_latent(
    z: Tensor,
    base_distribution: Any,
    ldj: Tensor,
    mq: Tensor,
    log_mix_wts: Tensor,
) -> Tensor:
    r"""Compute njNLL of given values"""
    # mu is zero where there is mask and sigma is 1 for the diagonal where there is mask
    _, _, K, M = base_distribution.U.shape
    if K <= M:
        neg_log_prob = multivariate_normal(z=z, mu=base_distribution.mean, sigma=base_distribution.cov, mq=mq)
    else:
        neg_log_prob = multivariate_normal_woodbury(z=z, mu=base_distribution.mean, U=base_distribution.U, mq=mq)
    comp_ldj = (ldj * mq[:, None, :]).sum(-1)
    comp_nll = neg_log_prob - comp_ldj
    total_nll = -torch.logsumexp(log_mix_wts - comp_nll, -1)
    mq_sum_safe = mq.sum(-1)
    mq_sum_safe[mq.sum(-1) == 0] = 1
    return (total_nll * (mq.sum(-1) > 0)) / mq_sum_safe


# normalized by the number of queries


def compute_mnll_on_latent(
    z: Tensor,
    base_distribution: Any,
    ldj: Tensor,
    mq: Tensor,
    log_mix_wts: Tensor,
) -> Tensor:
    """compute mnll given the z directly

    Args:
        z (Tensor: B, D, K): latent target variable
        base_distribution (Any): base multivariate guassian distribution
        ldj (Tensor: B, D, K): log determinent jacobian of splines applied on target variable
        mq (Tensor: B, K): mask for the quries
        log_mix_wts (Tensor: B, D): mixer weights for the components

    Returns:
        Tensor: mNLL
    """
    base_mean = base_distribution.mean
    base_stdev = torch.diagonal(base_distribution.cov, dim1=-2, dim2=-1) ** 0.5
    dist = torch.distributions.Normal(base_mean, base_stdev)
    # log_prob has shape [B, D, K]
    neg_log_prob = -dist.log_prob(z) * mq[:, None, :]

    comp_loss = neg_log_prob - ldj * mq[:, None, :]  # loglikelihood for each component

    mnll = -torch.logsumexp(log_mix_wts[:, :, None] - comp_loss, -2)  # logsumexp trick p(y) = w1p_1(y) + ... + wDp_D(y)
    return mnll * mq


def compute_robust_mean(yhat):
    """compute robustified mean: mean after removing the outliers"""
    yhat = torch.clamp(yhat, min=-10, max=10)  # set the values max and min to 10 and -10
    yhat = yhat.detach().cpu().numpy()
    ybar = stats.trim_mean(yhat, 0.1, axis=-1)  # compute mean after removing 10% of (tail) distribution
    ybar = torch.tensor(ybar)
    return ybar


def compute_likelihood_losses(model: Any, y: Tensor, mq: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute likelihood losses mnll and njnll

    Args:
        model (Any): Moses model
        y (Tensor): target values [B, K]
        mq (Tensor): mask for the targets [B, K]

    Returns:
        Tuple[Tensor, Tensor]: [njnll, mnll] for the batch given; shape [[1],[1]]
    """
    y_ = y.unsqueeze(1).repeat(1, model.num_components, 1)  # incorporate the dimension for moses components
    z, ldj = model._apply_flows_forward(y_)
    z *= mq[:, None, :]
    ldj *= mq[:, None, :]  # z \in [B, K, D], ldj \in [B, D]
    log_mix_wts = model.log_mixture_weights
    mnll = compute_mnll_on_latent(z, model.base_distribution, ldj, mq, log_mix_wts) * mq
    njnll = compute_njnll_on_latent(z, model.base_distribution, ldj, mq, log_mix_wts)
    n_instances = (mq.sum(-1) > 0).sum()
    njnll_mean = (njnll * (mq.sum(-1) > 0)).sum() / (n_instances)
    return njnll_mean, mnll.sum() / mq.sum()


def compute_njnll(model: Any, y: Tensor, mq: Tensor) -> Tuple[Tensor, Tensor]:
    y_ = y.unsqueeze(1).repeat(1, model.num_components, 1)  # incorporate the dimension for moses components
    z, ldj = model._apply_flows_forward(y_)
    z *= mq[:, None, :]
    ldj *= mq[:, None, :]  # z \in [B, K, D], ldj \in [B, D]
    log_mix_wts = model.log_mixture_weights
    njnll = compute_njnll_on_latent(z, model.base_distribution, ldj, mq, log_mix_wts)
    n_instances = (mq.sum(-1) > 0).sum()
    return (njnll * (mq.sum(-1) > 0)).sum() / (n_instances)


class AdditionalLossMetrics:
    """Container for computed loss metrics."""

    mse: Optional[Tensor] = None
    robust_mse: Optional[Tensor] = None
    crps: Optional[Tensor] = None
    energy_score: Optional[Tensor] = None

    def to_dict(self) -> Dict[str, Optional[Tensor]]:
        """Convert metrics to dictionary format."""
        return {
            "mse": self.mse,
            "robust_mse": self.robust_mse,
            "crps": self.crps,
            "energy_score": self.energy_score,
        }


class AdditionalLossesComputer:
    """
    Computes additional loss metrics for probabilistic models.

    This class calculates various loss metrics including MSE, robust MSE,
    CRPS (Continuous Ranked Probability Score), and Energy Score.
    """

    def __init__(self):
        """Initialize the loss computer with empty metrics."""
        self.metrics = AdditionalLossMetrics()

    def compute_additional_metrics(self, model: Any, y: Tensor, mq: Tensor, n_samples: int = 1000) -> AdditionalLossMetrics:
        """
        Compute all loss metrics for the given model predictions.

        Args:
            model: Probabilistic model with a samples() method
            y: Ground truth targets [B, K]
            mq: Model quantiles or other reference [B, K]
            n_samples: Number of samples to draw from the model

        Returns:
            LossMetrics: Container with all computed metrics

        Raises:
            ValueError: If inputs have incompatible shapes
            AttributeError: If model doesn't have required methods
        """
        # Validate inputs

        if y.shape != mq.shape:
            raise ValueError(f"Shape mismatch: y.shape={y.shape}, mq.shape={mq.shape}")
        # Generate samples from model: [B, n_samples, K]
        yhat = model.sample_joint(mq=mq, num_samples=n_samples)
        # Compute different estimators
        yhat_mean = yhat.mean(dim=-1)  # Standard mean
        yhat_robust_mean = compute_robust_mean(yhat).to(y.device)  # Robust estimator

        # Calculate all metrics
        self.metrics.mse = compute_mse(yhat_mean, y, mq)
        self.metrics.robust_mse = compute_mse(yhat_robust_mean, y, mq)
        self.metrics.crps = compute_crps(yhat, y, mq)
        self.metrics.energy_score = compute_energy_score(yhat, y, mq)

        return self.metrics
