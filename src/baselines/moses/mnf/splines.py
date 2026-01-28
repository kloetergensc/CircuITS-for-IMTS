import pdb

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .lr_splines import unconstrained_rational_linear_spline
from .rqs import unconstrained_RQS


class rlsplines(nn.Module):
    r"""rational linear splines"""

    def __init__(self, n_hiddens: int, n_gaussians: int, K: int = 16, B: int = 20):
        super(rlsplines, self).__init__()
        self.K = K
        self.B = B
        # self.w_param = nn.Linear(n_gaussians * n_hiddens, n_gaussians * K)
        # self.h_param = nn.Linear(n_gaussians * n_hiddens, n_gaussians * K)
        # self.d_param = nn.Linear(n_gaussians * n_hiddens, n_gaussians * (K - 1))
        # self.lda_param = nn.Linear(n_gaussians * n_hiddens, n_gaussians * K)
        self.w_param = nn.Linear(n_hiddens, K)
        self.h_param = nn.Linear(n_hiddens, K)
        self.d_param = nn.Linear(n_hiddens, K - 1)
        self.lda_param = nn.Linear(n_hiddens, K)

    def forward(
        self, z: Tensor, x: Tensor, inverse: bool = False
    ) -> tuple[Tensor, Tensor]:
        r"""compute the non-linear transformation of rational linear splines"""
        # pdb.set_trace()
        b_size, n_gauss, seq_len, _ = x.size()
        # x = x.permute(0, 2, 1, 3)
        # x = x.reshape(b_size, seq_len, -1)
        if not inverse:

            # W = self.w_param(x).permute(0, 2, 1)
            # H = self.h_param(x).permute(0, 2, 1)
            # D = self.d_param(x).permute(0, 2, 1)
            # lda = self.lda_param(x).permute(0, 2, 1)

            # W = W.reshape(b_size, n_gauss, seq_len, -1)
            # H = H.reshape(b_size, n_gauss, seq_len, -1)
            # D = D.reshape(b_size, n_gauss, seq_len, -1)
            # lda = lda.reshape(b_size, n_gauss, seq_len, -1)

            W = self.w_param(x)
            H = self.h_param(x)
            D = self.d_param(x)
            lda = self.lda_param(x)
            W, H = torch.softmax(W, dim=-1), torch.softmax(H, dim=-1)
            # W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            L = torch.sigmoid(lda)
            z, ld = unconstrained_rational_linear_spline(
                z, W, H, D, L, inverse=False, tail_bound=self.B
            )
            return z, ld
        else:
            # W = self.w_param(x).permute(0, 2, 1)
            # H = self.h_param(x).permute(0, 2, 1)
            # D = self.d_param(x).permute(0, 2, 1)
            # lda = self.lda_param(x).permute(0, 2, 1)

            # W = W.reshape(b_size, n_gauss, seq_len, -1)
            # H = H.reshape(b_size, n_gauss, seq_len, -1)
            # D = D.reshape(b_size, n_gauss, seq_len, -1)
            # lda = lda.reshape(b_size, n_gauss, seq_len, -1)

            W = self.w_param(x)
            H = self.h_param(x)
            D = self.d_param(x)
            lda = self.lda_param(x)

            W, H = torch.softmax(W, dim=-1), torch.softmax(H, dim=-1)
            # W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            L = torch.sigmoid(lda)
            z, ld = unconstrained_rational_linear_spline(
                z, W, H, D, L, inverse=True, tail_bound=self.B
            )
            return z, ld


class rqsplines(nn.Module):

    def __init__(self, n_gaussians, n_inputs, K=16, B=20):
        super(rqsplines, self).__init__()
        self.K = K
        self.B = B
        # self.W = nn.Parameter(torch.randn(n_gaussians, n_inputs, K))
        # self.H = nn.Parameter(torch.randn(n_gaussians, n_inputs, K))
        # self.D = nn.Parameter(torch.randn(n_gaussians, n_inputs, K - 1))

    def forward(self, x, w, h, d, inverse=False):
        if not inverse:

            W = self.W[None, :, :, :].repeat(x.shape[0], 1, 1, 1)
            H = self.H[None, :, :, :].repeat(x.shape[0], 1, 1, 1)
            D = self.D[None, :, :, :].repeat(x.shape[0], 1, 1, 1)
            W, H = torch.softmax(W, dim=-1), torch.softmax(H, dim=-1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            z, ld = unconstrained_RQS(x, W, H, D, inverse=False, tail_bound=self.B)
            return z, ld
        else:
            W = self.W[None, :, :, :].repeat(x.shape[0], 1, 1, 1)
            H = self.H[None, :, :, :].repeat(x.shape[0], 1, 1, 1)
            D = self.D[None, :, :, :].repeat(x.shape[0], 1, 1, 1)
            W, H = torch.softmax(W, dim=-1), torch.softmax(H, dim=-1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            z, ld = unconstrained_RQS(x, W, H, D, inverse=True, tail_bound=self.B)
            return z, ld
