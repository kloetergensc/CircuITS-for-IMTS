import pdb

import numpy as np
import properscoring as ps
import torch
from scipy import stats
from scipy.spatial import cKDTree as KDTree
from torch import Tensor, nn


class compute_losses:
    r"""compute losses"""

    def __init__(self):
        super(compute_losses).__init__()
        self.nlloss = nn.GaussianNLLLoss(full=True, reduction="none")

    def mnll(self, z: Tensor) -> Tensor:
        r"""Compute NLL of latent distribution, Gaussian"""

        mean = torch.zeros_like(z)

        var = torch.ones_like(z)

        ll = self.nlloss(mean, z, var)

        return ll

    def nll(self, z: Tensor, mq: Tensor, ldj: Tensor, mw: Tensor) -> Tensor:
        r"""Compute loss"""

        m = mq.unsqueeze(1).repeat(1, z.shape[1], 1)

        mgnll = (
            self.mnll(z) * m
        )  # B x (G x T x D) marginal gaussian negative log likelihood

        gnl = mgnll.sum(-1) - ldj  # B x G gaussian negative log likelihood

        if mw is not None:
            gnl = -torch.log(mw + 1e-8) + gnl  # adding log of mixure weights
            # gnl = -torch.log(torch.ones_like(gnl) / gnl.shape[-1]) + gnl
        nll = -torch.logsumexp(-gnl, -1)  # B x G gaussian likelihoods of mixtures
        obs_sample = mq.sum(-1)
        obs_sample = torch.where(obs_sample > 0, obs_sample, 1.0)
        njnll = nll / obs_sample
        actual_samples = mq.sum(-1).bool()
        njnll_ = njnll * actual_samples
        total_nll = njnll_.sum() / actual_samples.sum()
        return total_nll

    def mse(self, y: Tensor, yhat: Tensor, mq: Tensor) -> Tensor:
        r"""compute mse between y and yhat, mq is the query mask"""

        sq_err = (y - yhat) ** 2
        n_obs = mq.sum()
        mse = sq_err.sum() / n_obs

        return mse


class compute_marg_loss:
    r"""compute losses"""

    def __init__(self):
        super(compute_marg_loss).__init__()
        self.nlloss = nn.GaussianNLLLoss(full=True, reduction="none")

    def mnll(self, z: Tensor) -> Tensor:
        r"""Compute NLL of latent distribution, Gaussian"""

        mean = torch.zeros_like(z)

        var = torch.ones_like(z)

        ll = self.nlloss(mean, z, var)

        return ll

    def marg_nll(self, z: Tensor, mq: Tensor, ldj: Tensor, mw: Tensor) -> Tensor:
        r"""Compute loss"""

        m = mq.unsqueeze(1).repeat(1, z.shape[1], 1)

        mgnll = (
            self.mnll(z) * m
        )  # B x (G x T x D) marginal gaussian negative log likelihood

        gnl = mgnll.sum(-1) - ldj  # B x G gaussian negative log likelihood

        if mw is not None:
            gnl = -torch.log(mw + 1e-8) + gnl  # adding log of mixure weights

        nll = -torch.logsumexp(-gnl, -1)  # B x G gaussian likelihoods of mixtures
        # pdb.set_trace()

        actual_samples = mq.sum(-1).bool()

        nll_ = nll * actual_samples

        mean_nll = nll_.sum() / mq.sum()

        return mean_nll

    def mse(self, y: Tensor, yhat: Tensor, mq: Tensor, z_origin, ldj) -> Tensor:
        r"""compute mse between y and yhat, mq is the query mask"""
        # pdb.set_trace()
        mgnll = self.mnll(z_origin) * (mq[:, None, :])
        mgnll_ = mgnll.sum(-1) - ldj
        inds = mgnll_.argmin(1)
        ybar = yhat[range(len(inds)), inds]
        ybar = torch.clamp(ybar, min=-10, max=10)
        ybar = yhat.mean(1)
        # ybar= yhat[:,0]
        sq_err = (y - ybar) ** 2
        sq_err *= mq
        n_obs = mq.sum()
        mse = sq_err.sum() / n_obs
        return mse

    def robust_mse(self, y: Tensor, yhat: Tensor, mq: Tensor) -> Tensor:
        r"""compute mse between y and yhat, mq is the query mask"""
        yhat = torch.clamp(yhat, min=-10, max=10)
        yhat = yhat.detach().cpu().numpy()
        ybar = stats.trim_mean(yhat, 0.1, axis=1)
        ybar = torch.tensor(ybar).to(y.device)
        sq_err = (y - ybar) ** 2
        sq_err *= mq
        n_obs = mq.sum()
        mse = sq_err.sum() / n_obs
        return mse

    def med_mse(self, y: Tensor, yhat: Tensor, mq: Tensor) -> Tensor:
        r"""compute mse between y and yhat, mq is the query mask"""
        yhat = torch.clamp(yhat, min=-10, max=10)
        ybar = yhat[:, 0]
        sq_err = (y - ybar) ** 2
        sq_err *= mq
        n_obs = mq.sum()
        mse = sq_err.sum() / n_obs
        return mse

    def min_mse(self, y: Tensor, yhat: Tensor, mq: Tensor) -> Tensor:
        r"""compute mse from median between y and yhat, mq is the query mask"""
        yhat = torch.clamp(yhat, min=-10, max=10)
        sq_err = (y[:, None, :] - yhat) ** 2
        sq_err = sq_err.sum(-1)
        sq_err = sq_err.min(-1)[0]
        n_obs = mq.sum()
        mse = sq_err.sum() / n_obs
        return mse

    def crps(self, y: Tensor, yhat: Tensor, mq: Tensor) -> Tensor:
        yhat = torch.clamp(yhat, min=-10, max=10)
        y = y.detach().cpu().numpy()
        yhat = yhat.permute(0, 2, 1).detach().cpu().numpy()
        score = ps.crps_ensemble(y, yhat)
        score *= mq.detach().cpu().numpy()
        return score.sum() / mq.sum()

    def energy_score(
        self, y: Tensor, yhat: Tensor, mq: Tensor, beta: float = 1.0
    ) -> Tensor:

        nsamples = yhat.shape[1]

        y = y * mq
        yhat = yhat * (mq[:, None, :].repeat(1, nsamples, 1))
        yhat = torch.clamp(yhat, min=-10, max=10)

        # The Frobenius norm of a matrix is equal to the Euclidean norm of its element:
        # the square root of the sum of the square of its elements

        norm = torch.cdist(yhat, y[:, None, :], p=2).sum((1, 2)) / nsamples
        first_term = (norm**beta).sum()
        second_term = (
            -1.0
            / (2 * nsamples * (nsamples - 1))
            * torch.sum(
                torch.sum(
                    torch.pow(
                        torch.cdist(
                            yhat,
                            yhat,
                            p=2,
                        ),
                        beta,
                    ),
                    axis=(1, 2),
                ),
                dim=0,
            )
        )
        energy = first_term + second_term
        actual_samples = mq.sum(-1).bool().sum()
        return energy / actual_samples

    def energy_score_normalized_fast(
        self, y: Tensor, yhat: Tensor, mq: Tensor, beta: float = 1.0
    ):
        nsamples = yhat.shape[1]
        y = y * mq
        yhat = yhat * (mq[:, None, :].repeat(1, nsamples, 1))
        yhat = torch.clamp(yhat, min=-10, max=10)
        # The Frobenius norm of a matrix is equal to the Euclidean norm of its element:
        # the square root of the sum of the square of its elements
        diff = (yhat - y[:, None, :]) ** 2
        diff_sum = torch.sqrt(diff.sum(-1))
        first_term_batch = diff_sum.sum(-1) / nsamples

        second_term_batch = (
            -1.0
            / (2 * nsamples * (nsamples - 1))
            * torch.sum(
                torch.pow(
                    torch.cdist(
                        yhat,
                        yhat,
                        p=2,
                    ),
                    beta,
                ),
                axis=(1, 2),
            )
        )
        energy = first_term_batch + second_term_batch
        obs_sample = mq.sum(-1)
        obs_sample = torch.where(obs_sample > 0, obs_sample, 1.0)
        noramlized_energy = energy / obs_sample
        actual_samples = mq.sum(-1).bool()
        noramlized_energy_ = noramlized_energy * actual_samples
        return noramlized_energy_.sum() / actual_samples.sum()


def KLdivergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.

      Parameters
      ----------
      x : 2D array (n,d)
        Samples from distribution P, which typically represents the true
        distribution.
      y : 2D array (m,d)
        Samples from distribution Q, which typically represents the approximate
        distribution.

      Returns
      -------
      out : float
        The estimated Kullback-Leibler divergence D(P||Q).

      References
      ----------
      Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
    """

    # Check the dimensions are consistent
    x = x[:, None]
    y = y[:, None]
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n, d = x.shape
    m, dy = y.shape

    assert d == dy

    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=0.01, p=2)[0][:, 1]
    s = ytree.query(x, k=1, eps=0.01, p=2)[0]

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    return -np.log(r / s).sum() * d / n + np.log(m / (n - 1.0))


def wasserstein_distance(x, y):
    x_sorted = x.sort(-1)[0]
    y_sorted = y.sort(-1)[0]
    dist = (x_sorted - y_sorted) ** 2
    dist_mean = (dist.mean(-1)) ** 0.5
    return dist_mean
