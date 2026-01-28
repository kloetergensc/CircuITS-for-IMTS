import pdb

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .encoder_2 import encoder
from .functions import compute_gaussian_parameters
from .splines import rlsplines


class MarginalNormalizingFlows(nn.Module):
    """
    Implements a Marginalizable Normalizing Flow
    """

    def __init__(
        self,
        n_inputs: int = 5,
        n_hiddens: int = 128,
        n_flayers: int = 3,
        n_gaussians: int = 2,
        use_cov: bool = False,
        use_activation: bool = False,
        n_heads: int = 2,
        device="cpu",
    ):
        """
        Constructor.
        :param n_inputs: number of features in input
        :param n_hiddens: number of hidden units for each flow
        :param n_gaussians: number of gaussians in mixture of gaussians
        """
        super(MarginalNormalizingFlows, self).__init__()

        # save input arguments
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.use_cov = use_cov
        self.n_gaussians = n_gaussians
        self.n_flayers = n_flayers
        self.device = device
        self.encoder = encoder(n_inputs, n_hiddens, n_gaussians, n_heads)
        self.splines = nn.ModuleList()
        self.gm_param = compute_gaussian_parameters(n_hiddens, n_gaussians)
        for i in range(n_flayers):
            self.splines += [rlsplines(n_hiddens, n_gaussians)]

    def nf(
        self,
        obs: Tensor,
        mobs: Tensor,
        xq: Tensor,
        mq: Tensor,
        y: Tensor,
        inverse: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""flow model"""

        x, mw = self.encoder(
            obs, mobs, xq, mq
        )  # get the conditioning from the encoder, x: BxGxK

        mq = mq.unsqueeze(1).repeat(1, self.n_gaussians, 1)

        if not inverse:
            z = y.unsqueeze(1).repeat(1, self.n_gaussians, 1)

            ldj = torch.zeros_like(z).sum(-1)

            for i in range(self.n_flayers):
                z, ldj_splines = self.splines[i](z, x, inverse=False)
                ldj_splines = ldj_splines * mq
                ldj += ldj_splines.sum(-1)

            z *= mq
            mean, cov, _ = self.gm_param(x, mq)
            if self.use_cov:
                z = z - mean
                l = torch.linalg.cholesky(cov)
                z = torch.linalg.solve_triangular(l, z[:, :, :, None], upper=False)
                z = z[:, :, :, 0].clone()
                # pdb.set_trace()
                log_det_l = torch.log(torch.diagonal(l, dim1=-2, dim2=-1))
                ldj -= log_det_l.sum(
                    -1
                )  # sub the det of l instead of add the det of l^-1
            z *= mq
            if z.isnan().any():
                pdb.set_trace()
            return z, ldj, mw

        else:
            z = torch.randn(x.shape).to(x.device)
            ldj = torch.zeros_like(z).sum(-1)
            if self.use_cov:
                mean, cov, _ = self.gm_param(x, mq)
                l = torch.linalg.cholesky(cov)
                z = torch.matmul(l, z.unsqueeze(-1)).squeeze(-1)
                ldj -= torch.log(torch.diagonal(l, dim1=-2, dim2=-1)).sum(-1)
                z = z + mean
                z *= mq

            else:
                mw = None

                z, ldj_splines = self.splines[i](z, x, inverse=True)
                ldj_splines = ldj_splines * mq
                ldj -= ldj_splines.sum(-1)

            z *= mq
            inds = self.find_inds(mw)
            z, ldj = self.gather(z, ldj, inds)
            return z, ldj, mw

    def find_inds(self, mw: Tensor) -> Tensor:
        r"""find the component indices of the mixture"""

        distribution = torch.distributions.Categorical(mw)
        inds = distribution.sample()

        return inds

    def gather(self, x: Tensor, ldj: Tensor, inds: Tensor) -> tuple[Tensor, Tensor]:
        r"""gather only the components of the indices"""
        x = x[np.arange(x.shape[0]), inds]
        ldj = ldj[np.arange(ldj.shape[0]), inds]

        return x, ldj

    def forward(
        self,
        obs: Tensor,
        mobs: Tensor,
        xq: Tensor,
        mq: Tensor,
        y: Tensor,
        inverse: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""perform MNF"""

        if not inverse:
            z, ldj, mw = self.nf(obs, mobs, xq, mq, y, inverse=False)
            # pdb.set_trace()
            # y_, ldj_, mw_ = self.nf(obs, mobs, xq, mq, z, inverse=True)
            return z, ldj, mw

        else:
            y, ldj, mw = self.nf(obs=obs, mobs=mobs, xq=xq, mq=mq, y=None, inverse=True)
            return y, ldj, mw
