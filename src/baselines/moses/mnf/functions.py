import pdb

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class shiesh(nn.Module):
    slope = np.exp(1)
    threshold = 5

    def shiesh_(self, x: Tensor, inverse: bool = False) -> Tensor:
        """shiesh activation"""
        if not inverse:
            return torch.arcsinh(self.slope * torch.sinh(x))
        else:
            return torch.arcsinh(self.slope ** (-1) * torch.sinh(x))

    def Dshiesh(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """derivative of shiesh"""
        den = 1 + (self.slope * torch.sinh(x)) ** 2
        return self.slope * torch.cosh(x) / (den**0.5)

    def forward(self, x: Tensor, inverse: bool = False) -> tuple[Tensor, Tensor]:
        """compute shiesh and its derivative"""
        if not inverse:
            mask = x.abs() <= self.threshold
            x_out = torch.zeros_like(x)
            x_out[mask] = self.shiesh_(x[mask])
            x_out[~mask] = x[~mask] + torch.sign(x[~mask])
            dj = torch.ones_like(x)
            dj[mask] = self.Dshiesh(x[mask])
            return x_out, torch.log(dj)
        else:
            mask = x.abs() <= self.threshold - 1
            x_out = torch.zeros_like(x)
            x_out[mask] = self.shiesh_(x[mask], inverse=True)
            x_out[~mask] = x[~mask] - torch.sign(x[~mask])
            dj = torch.ones_like(x)
            dj[mask] = self.Dshiesh(x_out[mask])
            return x_out, -torch.log(dj)


class compute_gaussian_parameters(nn.Module):
    r"""compute the parameters of the mixure of gaussians"""

    def __init__(self, n_hiddens: int, n_gaussians: int):
        super(compute_gaussian_parameters, self).__init__()
        self.mean_parameters = nn.Linear(n_hiddens, 1)
        self.query_parameters = nn.Linear(n_hiddens, n_hiddens)
        self.key_parameters = nn.Linear(n_hiddens, n_hiddens)
        self.mixure_parameters = nn.Parameter(torch.randn(1, n_gaussians))
        self.value_parameters = nn.Linear(n_hiddens, 1)
        self.n_hiddens = n_hiddens

    def cov(self, x: Tensor, mq: Tensor) -> Tensor:
        r"""compute covariance"""
        attn_query = self.query_parameters(x)
        # attn_key = self.key_parameters(x)
        # attn_diag = self.value_parameters(x)
        id_tensor = torch.eye(x.shape[-2]).to(x.device)
        id_tensor = id_tensor[None, None, :, :].repeat(x.shape[0], x.shape[1], 1, 1)
        scores = torch.matmul(attn_query, attn_query.transpose(-2, -1)) / np.sqrt(
            self.n_hiddens
        )
        attn_mask = torch.matmul(mq.unsqueeze(-1), mq.unsqueeze(-2))
        cov = scores + id_tensor
        cov *= attn_mask
        cov += (1 - attn_mask) * id_tensor  # make diagonal entries with 0 to 1
        return cov

    def compute_mixure_weights(self, x: Tensor) -> Tensor:
        r"""compute mixure weights"""
        x = x.sum(-2)
        mw = self.mixure_parameters(x)
        return mw.squeeze(-1)

    def forward(self, x: Tensor, mq: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        r"output mean and covariance"
        mean = self.mean_parameters(x).squeeze(-1)
        cov = self.cov(x, mq)
        # mw = self.compute_mixure_weights(x)
        mw = (nn.Softmax(-1)(self.mixure_parameters)).repeat(x.shape[0], 1)
        return mean, cov, mw
