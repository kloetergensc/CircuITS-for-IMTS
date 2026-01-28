import pdb
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from . import grafiti_layers
from .change_rep import obs_rep


class GraFITi(nn.Module):

    def __init__(
        self,
        input_dim=41,
        attn_head=4,
        n_gaussians=2,
        latent_dim=128,
        n_layers=2,
        device="cuda",
    ):
        super().__init__()
        self.dim = input_dim  # input dimensions
        self.attn_head = attn_head  # no. of attention heads
        self.latent_dim = latent_dim  # latend dimension
        self.n_layers = n_layers  # number of grafiti layers
        self.device = device  # cpu or gpu
        self.n_gaussians = n_gaussians
        self.grafiti_ = grafiti_layers.grafiti_(
            self.dim,
            self.latent_dim,
            self.n_layers,
            self.attn_head,
            self.n_gaussians,
            device=device,
        )  # applying grafiti

    def forward(self, tx, cx, mx, x, tq, cq, mq):
        """
        Forward pass of the GraFITi model.
        Parameters:
        x_time: Tensor - Time points of the observations.
        x_vals: Tensor - Values of the observations.
        x_mask: Tensor - Mask for the observations.
        y_mask: Tensor - Mask for the queries.
        Returns:
        h: Tensor - Output of the GraFITi model; conditioning module for profiti model.
        """
        x_time, x_vals, x_mask, y_mask = obs_rep(tx, cx, mx, x, tq, cq, mq, self.dim)

        h_qry, h_obs = self.grafiti_(x_time, x_vals, x_mask, y_mask)
        return (
            h_obs,
            h_qry.reshape(
                h_qry.shape[0], h_qry.shape[1], self.n_gaussians, self.latent_dim
            ).permute(0, 2, 1, 3),
        )
