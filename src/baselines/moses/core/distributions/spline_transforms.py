import pdb
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_linear_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    unnormalized_lambdas,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):

    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = (
        rational_linear_spline(
            inputs=inputs[inside_interval_mask],
            unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
            unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
            unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
            unnormalized_lambdas=unnormalized_lambdas[inside_interval_mask, :],
            inverse=inverse,
            left=-tail_bound,
            right=tail_bound,
            bottom=-tail_bound,
            top=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )
    )

    return outputs, logabsdet


def rational_linear_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    unnormalized_lambdas,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left

    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    lambdas = 0.95 * torch.sigmoid(unnormalized_lambdas) + 0.025

    lam = lambdas.gather(-1, bin_idx)[..., 0]
    wa = 1
    wb = torch.sqrt(input_derivatives / input_derivatives_plus_one) * wa
    wc = (
        lam * wa * input_derivatives + (1 - lam) * wb * input_derivatives_plus_one
    ) / input_delta
    ya = input_cumheights
    yb = input_heights + input_cumheights
    yc = ((1 - lam) * wa * ya + lam * wb * yb) / ((1 - lam) * wa + lam * wb)

    if inverse:

        numerator = (lam * wa * (ya - inputs)) * (inputs <= yc).float() + (
            (wc - lam * wb) * inputs + lam * wb * yb - wc * yc
        ) * (inputs > yc).float()

        denominator = ((wc - wa) * inputs + wa * ya - wc * yc) * (
            inputs <= yc
        ).float() + ((wc - wb) * inputs + wb * yb - wc * yc) * (inputs > yc).float()

        theta = numerator / denominator

        outputs = theta * input_bin_widths + input_cumwidths

        derivative_numerator = (
            wa * wc * lam * (yc - ya) * (inputs <= yc).float()
            + wb * wc * (1 - lam) * (yb - yc) * (inputs > yc).float()
        ) * input_bin_widths

        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(abs(denominator))

        return outputs, logabsdet
    else:

        theta = (inputs - input_cumwidths) / input_bin_widths

        numerator = (wa * ya * (lam - theta) + wc * yc * theta) * (
            theta <= lam
        ).float() + (wc * yc * (1 - theta) + wb * yb * (theta - lam)) * (
            theta > lam
        ).float()

        denominator = (wa * (lam - theta) + wc * theta) * (theta <= lam).float() + (
            wc * (1 - theta) + wb * (theta - lam)
        ) * (theta > lam).float()

        outputs = numerator / denominator

        derivative_numerator = (
            wa * wc * lam * (yc - ya) * (theta <= lam).float()
            + wb * wc * (1 - lam) * (yb - yc) * (theta > lam).float()
        ) / input_bin_widths

        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(abs(denominator))

        return outputs, logabsdet


class RationalLinearSplineFlow(nn.Module):
    """
    Rational Linear Splines transformation layer for normalizing flows.

    This module implements a learnable rational linear spline transformation
    that can be used as a coupling layer in normalizing flows. The transformation
    is conditioned on input features and provides both forward and inverse
    operations with tractable Jacobian determinants.

    The rational linear splines offer several advantages:
    - Smooth and differentiable transformations
    - Flexible modeling capacity with configurable number of bins
    - Numerically stable forward and inverse operations
    - Efficient computation of log-determinants

    Args:
        d_model: Dimension of conditioning features
        num_bins: Number of spline bins (default: 16)
        tail_bound: Boundary for spline domain (default: 20.0)
        min_bin_width: Minimum bin width for stability
        min_bin_height: Minimum bin height for stability
        min_derivative: Minimum derivative for stability
    """

    def __init__(
        self,
        d_model: int,
        num_bins: int = 16,
        tail_bound: float = 20.0,
        min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
        min_derivative: float = DEFAULT_MIN_DERIVATIVE,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        # Parameter networks for spline components
        self.width_net = nn.Linear(d_model, num_bins)
        self.height_net = nn.Linear(d_model, num_bins)
        self.derivative_net = nn.Linear(
            d_model, num_bins - 1
        )  # +1 for boundary derivatives
        self.lambda_net = nn.Linear(d_model, num_bins)
        # self.offset_layer = nn.Linear(d_model, 1)
        self.widths = None
        self.heights = None
        self.derivatives = None
        self.lambdas = None

    def forward(self, conditioning: Tensor):
        """
        Compute spline parameters from conditioning features.

        Args:
            conditioning: Conditioning features [*, d_model]

        Returns:
            Tuple of (widths, heights, derivatives, lambdas)
        """
        widths = self.width_net(conditioning)
        heights = self.height_net(conditioning)
        derivatives = self.derivative_net(conditioning)
        lambdas = self.lambda_net(conditioning)

        self.widths = torch.softmax(widths, dim=-1)
        self.heights = torch.softmax(heights, dim=-1)

        self.derivatives = F.softplus(derivatives)
        self.lambdas = torch.sigmoid(lambdas)

        # self.offset = self.offset_layer(conditioning).squeeze(-1)

    def forward_(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward transformation through rational linear splines.

        Args:
            inputs: Input values to transform [*, 1]
            conditioning: Conditioning features [*, d_model]

        Returns:
            Tuple of (transformed_values, log_abs_det_jacobian)
        """
        # inputs = self.offset + inputs
        # Apply spline transformation
        outputs, log_abs_det = unconstrained_rational_linear_spline(
            inputs=inputs,
            unnormalized_widths=self.widths,
            unnormalized_heights=self.heights,
            unnormalized_derivatives=self.derivatives,
            unnormalized_lambdas=self.lambdas,
            inverse=False,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        return outputs, log_abs_det

    def inverse_(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Inverse transformation through rational linear splines.

        Args:
            inputs: Input values to transform [*, 1]
            conditioning: Conditioning features [*, d_model]

        Returns:
            Tuple of (inverse_transformed_values, log_abs_det_jacobian)
        """
        if len(inputs.shape) > 3:  # this happens during sampling
            nsamples = inputs.shape[-1]
            # offset = self.offset.unsqueeze(-1)
            widths = self.widths.unsqueeze(-2).repeat(1, 1, 1, nsamples, 1)
            heights = self.heights.unsqueeze(-2).repeat(1, 1, 1, nsamples, 1)
            derivatives = self.derivatives.unsqueeze(-2).repeat(1, 1, 1, nsamples, 1)
            lambdas = self.lambdas.unsqueeze(-2).repeat(1, 1, 1, nsamples, 1)
        else:
            # offset = self.offset
            widths = self.widths
            heights = self.heights
            derivatives = self.derivatives
            lambdas = self.lambdas

        # inputs = inputs - offset
        # Apply inverse spline transformation
        outputs, log_abs_det = unconstrained_rational_linear_spline(
            inputs=inputs,
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            unnormalized_lambdas=lambdas,
            inverse=True,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        return outputs, log_abs_det
