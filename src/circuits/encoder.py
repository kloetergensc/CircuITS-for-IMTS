import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.baselines.moses.core.encoders.transformer import MultiHeadAttention


class TimeEmbedding(nn.Module):
    """
    Generates sinusoidal embeddings for continuous scalar timestamps.
    """

    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(1, d)

    def forward(self, timestamps):
        # (..., 1) * (D/2, ) -> (..., D/2)
        return torch.sin(self.linear(timestamps.unsqueeze(-1)))


class CircuitsEncoder(nn.Module):
    def __init__(self, num_channels, d_hidden, output_dim, components, nhead=4, dropout=0.0):
        super().__init__()
        self.num_channels = num_channels
        self.d_hidden = d_hidden
        self.components = components

        # --- 1. Embedding Components ---
        self.time_embedder = TimeEmbedding(d_hidden)
        self.channel_embedder = nn.Embedding(num_channels, d_hidden)
        self.input_compressor = nn.Linear(d_hidden + d_hidden + 1, d_hidden)

        # --- 2. Intra-Channel Aggregation (Cross Attention) ---
        self.channel_aggregator = nn.MultiheadAttention(embed_dim=d_hidden, num_heads=nhead, batch_first=True, dropout=dropout)

        # --- 3. Inter-Channel Update (Self Attention) ---
        self.inter_channel_update = nn.MultiheadAttention(embed_dim=d_hidden, num_heads=nhead, batch_first=True, dropout=dropout)
        self.norm_inter = nn.LayerNorm(d_hidden)

        # --- 4. Weight Generation Heads (New Features) ---

        # A. Final Weights: Target shape (BS, components)
        # We learn 1 query vector to extract this global info
        self.final_weight_query = nn.Parameter(torch.randn(1, 1, d_hidden))
        self.final_weight_attn = nn.MultiheadAttention(embed_dim=d_hidden, num_heads=nhead, batch_first=True, dropout=dropout)
        self.final_weight_head = nn.Linear(d_hidden, components)

        # B. Sum Layer Weights: Target shape (BS, C-1, components, components*2)
        # We learn (C-1) query vectors.
        self.sum_layer_query = nn.Parameter(torch.randn(1, num_channels - 1, d_hidden))
        self.sum_layer_attn = nn.MultiheadAttention(embed_dim=d_hidden, num_heads=nhead, batch_first=True, dropout=dropout)
        self.sum_layer_head = nn.Linear(d_hidden, components**3)

        # --- 5. Main Inference Output ---
        self.final_head = nn.Linear(d_hidden + d_hidden, output_dim)
        self.nhead = nhead

    def forward(self, obs, mobs, xq, mq):
        """
        Args:
            obs: (BS, T, 3)
            mobs: (BS, T)
            xq: (BS, Q, 2)
            mq: (BS, Q)
        Returns:
            output: (BS, Q, E)
            final_weights: (BS, components)
            sum_layer_weights: (BS, C-1, components, components*2)
        """
        BS, T, _ = obs.shape
        BS, Q, _ = xq.shape
        device = obs.device

        # ==========================================================
        # Step 1: Embed Observations
        # ==========================================================
        obs_time = obs[:, :, 0]
        obs_chan = obs[:, :, 1].long()
        obs_val = obs[:, :, 2].unsqueeze(-1)

        e_time = self.time_embedder(obs_time)
        e_chan = self.channel_embedder(obs_chan)

        triplet_concat = torch.cat([e_time, obs_val, e_chan], dim=-1)
        obs_embeds = self.input_compressor(triplet_concat)

        # ==========================================================
        # Step 2: Aggregate triplets by Channel (Cross Attention)
        # ==========================================================
        all_channel_indices = torch.arange(self.num_channels, device=device).unsqueeze(0).repeat(BS, 1)
        channel_queries = self.channel_embedder(all_channel_indices)  # (BS, C, H)

        # Masking Logic
        obs_chan_expanded = obs_chan.unsqueeze(1)
        target_chan_expanded = all_channel_indices.unsqueeze(2)
        channel_match_mask = obs_chan_expanded == target_chan_expanded  # (BS, C, T)
        valid_obs_mask = mobs.unsqueeze(1).to(bool)

        # MHA mask: True means Ignore
        attn_mask = ~(channel_match_mask & valid_obs_mask)
        is_row_fully_masked = attn_mask.all(dim=-1)  # (BS, C)
        if is_row_fully_masked.any() and attn_mask.shape[-1] > 0:
            attn_mask[:, :, 0].masked_fill_(is_row_fully_masked, False)
        attn_mask_heads = attn_mask.repeat_interleave(self.nhead, dim=0)
        channel_states, _ = self.channel_aggregator(query=channel_queries, key=obs_embeds, value=obs_embeds, attn_mask=attn_mask_heads)

        # Fix NaNs for empty channels
        has_data_mask = (channel_match_mask & valid_obs_mask).any(dim=-1)  # (BS, C)
        channel_states = channel_states * has_data_mask.unsqueeze(-1).float()

        # Replace empty channels with their static query embeddings
        channel_states = torch.where(has_data_mask.unsqueeze(-1), channel_states, channel_queries)

        # ==========================================================
        # Step 3: Inter-Channel Update (Self Attention)
        # ==========================================================
        # All channels (including the statically filled ones) attend to each other
        updated_states, _ = self.inter_channel_update(query=channel_states, key=channel_states, value=channel_states)
        channel_states = self.norm_inter(channel_states + updated_states)

        # ==========================================================
        # Step 4: Generate Auxiliary Weights
        # ==========================================================

        # --- A. Final Weights (BS, components) ---
        # Expand latent query to batch size: (BS, 1, H)
        f_query = self.final_weight_query.expand(BS, -1, -1)

        # Cross Attention: Query=Latent, componentsey/Val=ChannelStates
        f_ctx, _ = self.final_weight_attn(query=f_query, key=channel_states, value=channel_states)

        # Project and Remove Sequence Dim (BS, 1, components) -> (BS, components)
        final_weights = self.final_weight_head(f_ctx).squeeze(1)
        final_weights = torch.log_softmax(final_weights, dim=-1)

        # --- B. Sum Layer Weights (BS, C-1, components, components*2) ---
        # Expand latent query to batch size: (BS, C-1, H)
        s_query = self.sum_layer_query.expand(BS, -1, -1)

        # Cross Attention
        s_ctx, _ = self.sum_layer_attn(query=s_query, key=channel_states, value=channel_states)

        # Project: (BS, C-1, H) -> (BS, C-1, components^3)
        s_flat = self.sum_layer_head(s_ctx)

        # Reshape: (BS, C-1, components, components^2)
        sum_layer_weights = s_flat.view(BS, self.num_channels - 1, self.components**2, self.components)
        sum_layer_weights = torch.log_softmax(sum_layer_weights, dim=-2)

        # ==========================================================
        # Step 5: Main Inference on XQ
        # ==========================================================
        xq_time = xq[:, :, 0]
        xq_chan = xq[:, :, 1].long()

        xq_time_emb = self.time_embedder(xq_time)

        gather_index = xq_chan.unsqueeze(-1).expand(-1, -1, self.d_hidden)
        xq_chan_emb = torch.gather(channel_states, 1, gather_index)

        final_input = torch.cat([xq_time_emb, xq_chan_emb], dim=-1)
        output = self.final_head(final_input)

        if mq is not None:
            output = output * mq.unsqueeze(-1).float()

        return output, sum_layer_weights, final_weights


class TFEncoder(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_hiddens: int,
        n_heads: int = 2,
        components: int = 2,
    ):
        super(TFEncoder, self).__init__()

        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
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

        self.weight_query = nn.Parameter(torch.randn(1, n_inputs, n_hiddens))

        self.mha_self = MultiHeadAttention(n_hiddens, n_heads)
        self.mha_wts = MultiHeadAttention(n_hiddens, n_heads)
        self.mha_cross = MultiHeadAttention(n_hiddens, n_heads)

        self.sum_weight_linear = nn.Linear(n_hiddens, (components**3))
        self.final_weight_linear = nn.Linear(n_hiddens, components)
        self.components = components

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

        attn_mask_weight = mobs.unsqueeze(-2).repeat(1, self.n_inputs, 1)

        obs_embed = self.mha_self(obs_embed, obs_embed, obs_embed, attn_mask_self)
        obs_embed_non_act = obs_embed.clone()
        obs_embed = self.relu(obs_embed)

        qry_embed = self.mha_cross(qry_embed, obs_embed, obs_embed, attn_mask_cross)
        wgt_query = self.weight_query.repeat(xq.shape[0], 1, 1)
        wgt_attn = self.mha_wts(attn_queries=wgt_query, attn_keys=obs_embed, attn_vals=obs_embed, attn_mask=attn_mask_weight)

        sum_weights = F.log_softmax(self.sum_weight_linear(F.relu(wgt_attn[:, 1:, :])).reshape(-1, self.n_inputs - 1, self.components**2, self.components), dim=-2)

        final_weights = F.log_softmax(self.final_weight_linear(F.relu(wgt_attn[:, 0, :])), dim=-1)

        # if (mw < 1e-10).any():
        #     pdb.set_trace()
        return qry_embed, sum_weights, final_weights
