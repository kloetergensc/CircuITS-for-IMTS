"""
Copyright 2023 ServiceNow
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
from typing import Tuple

import torch
from torch import nn

from .flow import DeepSigmoidFlow

EPSILON = 1e-6


class BatchedLinear(nn.Module):
    def __init__(self, K, D_in, D_out):
        super().__init__()
        # Shape: (K, D_in, D_out)
        self.weight = nn.Parameter(torch.randn(K, D_in, D_out), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(K, D_out), requires_grad=True)
        self.D_in = D_in
        self.D_out = D_out

        self.reset_parameters()

    def reset_parameters(self):
        """
        Replicates the default initialization of nn.Linear
        adjusted for the extra K dimension.
        """
        # 1. Calculate 'fan_in' (input features)
        # We only care about D_in, not K, because the dot product happens
        # individually per k.
        fan_in = self.D_in

        # 2. Initialize Weights (Kaiming Uniform)
        # PyTorch default uses a=sqrt(5) for Linear layers
        a = math.sqrt(5)
        gain = torch.nn.init.calculate_gain("relu", a)
        std = gain / math.sqrt(fan_in)

        # Calculate the uniform bound: bound = sqrt(3) * std
        bound = math.sqrt(3.0) * std

        # Apply uniform distribution
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

        # 3. Initialize Bias
        # PyTorch initializes bias with Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))
        bias_bound = 1 / math.sqrt(fan_in)
        with torch.no_grad():
            self.bias.uniform_(-bias_bound, bias_bound)

    def forward(self, x):

        res = x @ self.weight.unsqueeze(0) + self.bias.unsqueeze(0).unsqueeze(-2)
        return res


class DSFMarginal(nn.Module):
    """
    Compute the marginals using a Deep Sigmoid Flow conditioned using a MLP.
    The conditioning MLP uses the embedding from the encoder as its input.
    """

    def __init__(
        self,
        context_dim: int,
        mlp_layers: int,
        mlp_dim: int,
        flow_layers: int,
        flow_hid_dim: int,
        no_logit: bool = True,
        K=0,
    ):
        """
        Parameters:
        -----------
        context_dim: int
            Size of the context (embedding created by the encoder) that will be sent to the conditioner.
        mlp_layers: int
            Number of layers for the conditioner MLP.
        mlp_dim: int
            Dimension of the hidden layers of the conditioner MLP.
        flow_layers: int
            Number of layers for the Dense Sigmoid Flow.
        flow_hid_dim: int
            Dimension of the hidden layers of the Dense Sigmoid Flow.
        no_logit: bool,
            if True the flow will only return values in from the range [0,1]
        """

        super().__init__()

        self.context_dim = context_dim
        self.mlp_layers = mlp_layers
        self.mlp_dim = mlp_dim
        self.flow_layers = flow_layers
        self.flow_hid_dim = flow_hid_dim

        self.marginal_flow = DeepSigmoidFlow(n_layers=self.flow_layers, hidden_dim=self.flow_hid_dim, no_logit=no_logit)

        if K == 0:
            elayers = [nn.Linear(self.context_dim, self.mlp_dim), nn.ReLU()]
            for _ in range(1, self.mlp_layers):
                elayers += [nn.Linear(self.mlp_dim, self.mlp_dim), nn.ReLU()]
            elayers += [nn.Linear(self.mlp_dim, self.marginal_flow.total_params_length)]
        else:
            elayers = [BatchedLinear(K=K, D_in=self.context_dim, D_out=self.mlp_dim), nn.ReLU()]
            for _ in range(1, self.mlp_layers):
                elayers += [BatchedLinear(K=K, D_in=self.mlp_dim, D_out=self.mlp_dim), nn.ReLU()]
            elayers += [BatchedLinear(K=K, D_in=self.mlp_dim, D_out=self.marginal_flow.total_params_length)]

        self.marginal_conditioner = nn.Sequential(*elayers)

    def forward_logdet(self, context: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the cumulative density function of a marginal conditioned using the given context, for the given value of x.
        Also returns the logarithm of the derivative of this transformation.

        Parameters:
        -----------
        context: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            The series and time steps dimensions are merged.
        x: Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            A tensor containing the value to be transformed using the CDF.
            The series and time steps dimensions are merged.
            If a third dimension is present, then the context is considered to be constant across this dimension.
        Returns:
        --------
        u: torch.Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            The CDF at the given point, a value between 0 and 1.
            The series and time steps dimensions are merged.
            The shape of the output is the same as the shape of x.
        logdet: torch.Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            The logarithm of the derivative of the transformation.
            The series and time steps dimensions are merged.
            The shape of the output is the same as the shape of x.
        """
        marginal_params = self.marginal_conditioner(context)
        # If x has both a variable and a sample dimension, then add a singleton dimension to marginal_params to have the correct shape
        if marginal_params.dim() == x.dim():
            marginal_params = marginal_params[:, :, None, :]

        return self.marginal_flow.forward(marginal_params, x)

    def forward_no_logdet(self, context: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the cumulative density function of a marginal conditioned using the given context, for the given value of x.

        Parameters:
        -----------
        context: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            The series and time steps dimensions are merged.
        x: Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            A tensor containing the value to be transformed using the CDF.
            The series and time steps dimensions are merged.
            If a third dimension is present, then the context is considered to be constant across this dimension.
        Returns:
        --------
        u: torch.Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            The CDF at the given point, a value between 0 and 1.
            The series and time steps dimensions are merged.
            The shape of the output is the same as the shape of x.
        """
        marginal_params = self.marginal_conditioner(context)  # [batch, series*timesteps, self.marginal_flow.total_params_length]
        # If x has both a variable and a sample dimension, then add a singleton dimension to marginal_params to have the correct shape
        if marginal_params.dim() == x.dim():
            marginal_params = marginal_params[:, :, None, :]

        return self.marginal_flow.forward_no_logdet(marginal_params, x)

    def inverse(
        self,
        context: torch.Tensor,
        u: torch.Tensor,
        max_iter: int = 100,
        precision: float = 1e-6,
        max_value: float = 30,
    ) -> torch.Tensor:
        """
        Compute the inverse cumulative density function of a marginal conditioned using the given context, for the given value of u.
        This method uses a dichotomic search.
        The gradient of this method cannot be computed, so it should only be used for sampling.

        Parameters:
        -----------
        context: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            The series and time steps dimensions are merged.
        u: Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            A tensor containing the value to be transformed using the inverse CDF.
            The series and time steps dimensions are merged.
            If a third dimension is present, then the context is considered to be constant across this dimension.
        max_iter: int, default = 100
            The maximum number of iterations for the dichotomic search.
            The precision of the result should improve by a factor of 2 at each iteration.
        precision: float, default = 1e-6
            If the difference between CDF(x) and u is less than this value for all variables, stop the search.
        max_value: float, default = 1000.0
            The absolute upper bound on the possible output.
        Returns:
        --------
        x: torch.Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            The inverse CDF at the given value.
            The series and time steps dimensions are merged.
            The shape of the output is the same as the shape of u.
        """
        marginal_params = self.marginal_conditioner(context)
        # If u has both a variable and a sample dimension, then add a singleton dimension to marginal_params to have the correct shape
        if marginal_params.dim() == u.dim():
            marginal_params = marginal_params.unsqueeze(-2)
        left = -max_value * torch.ones_like(u)
        right = max_value * torch.ones_like(u)
        for _ in range(max_iter):
            mid = (left + right) / 2
            error = self.marginal_flow.forward_no_logdet(marginal_params, mid) - u
            left[error <= 0] = mid[error <= 0]
            right[error >= 0] = mid[error >= 0]

            max_error = error.abs().max().item()
            if max_error < precision:
                break
        # Get the highest value smaller than the max_value
        # mask_max = mid < max_value
        # highest_below_max = mid[mask_max].max().item() if mask_max.any() else float("-inf")
        # mask_min = mid > -max_value
        # lowest_above_min = mid[mask_min].min().item() if mask_min.any() else float("inf")
        # mid = mid.clamp(min=lowest_above_min, max=highest_below_max)
        return mid
