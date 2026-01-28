import math
import pdb

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e6)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class multiheadattention(nn.Module):
    def __init__(self, n_hiddens: int, n_heads: int):
        super(multiheadattention, self).__init__()
        self.query_param = nn.Linear(n_hiddens, n_hiddens * n_heads)
        self.key_param = nn.Linear(n_hiddens, n_hiddens * n_heads)
        self.val_param = nn.Linear(n_hiddens, n_hiddens * n_heads)
        self.linear_o = nn.Linear(n_heads * n_hiddens, n_hiddens)
        self.n_hiddens = n_hiddens
        self.n_heads = n_heads

    def forward(
        self,
        attn_queries: Tensor,
        attn_keys: Tensor,
        attn_vals: Tensor,
        attn_mask: Tensor,
    ) -> Tensor:
        r"""compute mha"""

        q = self.query_param(attn_queries)
        k = self.query_param(attn_keys)
        v = self.query_param(attn_vals)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)

        if attn_mask is not None:
            mask = attn_mask.repeat(self.n_heads, 1, 1)

        x = ScaledDotProductAttention()(q, k, v, mask)
        x = self._reshape_from_batches(x)
        x = attn_queries + self.linear_o(x)
        return x

    def _reshape_to_batches(self, x: Tensor) -> Tensor:
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.n_heads
        return (
            x.reshape(batch_size, seq_len, self.n_heads, sub_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.n_heads, seq_len, sub_dim)
        )

    def _reshape_from_batches(self, x: Tensor) -> Tensor:
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.n_heads
        out_dim = in_feature * self.n_heads
        return (
            x.reshape(batch_size, self.n_heads, seq_len, in_feature)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, out_dim)
        )


class encoder(nn.Module):
    def __init__(
        self, n_inputs: int, n_hiddens: int, n_gaussians: int, n_heads: int = 2
    ):
        super(encoder, self).__init__()

        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n_gaussians = n_gaussians
        self.n_heads = n_heads

        self.qembed0 = nn.Linear(1, 1)
        self.qembed1 = nn.Linear(1, n_hiddens - 1)

        self.obsembed0 = nn.Linear(1, 1)
        self.obsembed1 = nn.Linear(1, n_hiddens - 1)

        self.cobs_embed = nn.Linear(n_inputs, n_hiddens)
        self.cq_embed = nn.Linear(n_inputs, n_hiddens)

        self.relu = nn.ReLU()

        self.query_param = nn.Linear(n_hiddens + n_hiddens, n_hiddens)
        self.key_param = nn.Linear(n_hiddens + n_hiddens + 1, n_hiddens)

        self.mha_self = multiheadattention(n_hiddens, n_heads)
        self.mha_wts = multiheadattention(n_hiddens, n_heads)
        self.mha_cross = multiheadattention(n_hiddens, n_heads)

        self.split_ngaussians = nn.Linear(n_hiddens, n_hiddens * n_gaussians)

        self.mixture_wt_param = nn.Parameter(torch.randn(n_gaussians, n_hiddens))
        self.mix_wts_nn = nn.Linear(n_hiddens, 1)
        # self.temperature = nn.Parameter(torch.ones(1)*(5))

    def forward(self, obs: Tensor, mobs: Tensor, xq: Tensor, mq: Tensor) -> Tensor:
        r"""compute the conditionings"""

        cq = xq[:, :, 1]
        tq = xq[:, :, 0:1]

        cobs = obs[:, :, 1]
        tobs = obs[:, :, 0:1]
        xobs = obs[:, :, 2:]

        tq_0 = self.qembed0(tq)
        tq_1 = torch.sin(self.qembed1(tq))
        tq = torch.cat((tq_0, tq_1), -1)

        tobs_0 = self.obsembed0(tobs)
        tobs_1 = torch.sin(self.obsembed1(tobs))
        tobs = torch.cat((tobs_0, tobs_1), -1)

        cobs = F.one_hot(cobs.to(torch.int64), num_classes=self.n_inputs).to(obs.dtype)
        cq = F.one_hot(cq.to(torch.int64), num_classes=self.n_inputs).to(obs.dtype)

        cobs = self.relu(self.cobs_embed(cobs))
        cq = self.relu(self.cq_embed(cq))

        obs_ = torch.cat((tobs, cobs, xobs), -1)
        xq_ = torch.cat((tq, cq), -1)

        obs_ *= mobs.unsqueeze(-1).repeat(1, 1, obs_.shape[-1])
        xq_ *= mq.unsqueeze(-1).repeat(1, 1, xq_.shape[-1])

        qry_embed = self.query_param(xq_)
        obs_embed = self.key_param(obs_)

        attn_mask_self = torch.matmul(mobs.unsqueeze(-1), mobs.unsqueeze(-2))

        attn_mask_cross = torch.matmul(mq.unsqueeze(-1), mobs.unsqueeze(-2))

        wts_mask = torch.ones_like(self.mixture_wt_param[:, 0:1])
        attn_mix_wts_mask = torch.matmul(
            wts_mask[None, :, :].repeat(qry_embed.shape[0], 1, 1), mobs.unsqueeze(-2)
        )
        wts_query = self.mixture_wt_param[None, :, :].repeat(qry_embed.shape[0], 1, 1)

        x = self.mha_self(obs_embed, obs_embed, obs_embed, attn_mask_self)
        x = self.relu(x)

        mw_ = self.mha_wts(wts_query, x, x, attn_mix_wts_mask)
        mw = self.mix_wts_nn(mw_).squeeze(-1)

        mw = nn.Softmax(-1)(mw)
        # mw = nn.Sigmoid()(mw)
        # mw = mw / mw.sum(axis=-1, keep_dims=True)

        x = self.mha_cross(qry_embed, x, x, attn_mask_cross)
        x = self.relu(x)
        x = self.split_ngaussians(x).reshape(
            -1, x.shape[1], self.n_gaussians, self.n_hiddens
        )

        # if (mw < 1e-10).any():
        #     pdb.set_trace()
        return x.permute(0, 2, 1, 3), mw
