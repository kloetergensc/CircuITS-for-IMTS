import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.multivariate_normal import MultivariateNormal

from src.circuits.encoder import CircuitsEncoder, TFEncoder
from src.circuits.gaussian_copula import compute_gaussian_parameters
from src.circuits.marginal import DSFMarginal as DSF
from src.circuits.sum_product_network import parallel_circuit
from src.utils.utils import compress_masked_tensor


class CircuITS(nn.Module):
    """
    Marginalizable Normalizing Flow (MNF) implementation.
    """

    def __init__(
        self,
        n_channels: int = 5,
        hidden_dim: int = 32,
        n_heads: int = 2,
        flow_args: dict = None,
        device: str = "cpu",
        components: int = 2,
        marginal_flow="dsf",
        encoder_type="default",
        channel_order=[],
    ):
        """
        Args:
            n_channels: Number of input features.
            hidden_dim: Number of hidden units per flow.
            n_flayers: Number of flow layers.
            n_gaussians: Number of Gaussians in mixture.
            use_cov: Whether to use covariance.
            use_activation: Whether to use activation.
            n_heads: Number of attention heads.
            copula_args: Arguments for AttentionalCopula.
            flow_args: Arguments for DSFMarginal.
            device: Device to run the model on.
        """
        super().__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.device = device

        if encoder_type == "TF":
            self.encoder = TFEncoder(n_inputs=n_channels, n_hiddens=hidden_dim * components, n_heads=n_heads, components=components)
        elif encoder_type == "default":
            self.encoder = CircuitsEncoder(num_channels=n_channels, d_hidden=hidden_dim, nhead=n_heads, components=components, output_dim=hidden_dim * components)
        # self.copula = AttentionalCopula(**(copula_args or {}))
        else:
            raise NotImplementedError("Encoder type can only be default or TF")
        self.components = components
        self.n_flayers = flow_args["flow_layers"]
        if marginal_flow == "dsf":
            self.base_distribution = "uniform"
            self.DSF = DSF(**(flow_args or {}), no_logit=True)
        elif marginal_flow == "none":
            self.base_distribution = "normal"
        else:
            raise NotImplementedError(' chose on of these marginal flows:  ["none", "dsf"]')

        self.gm_param = compute_gaussian_parameters(embedding_dim=hidden_dim, hidden_dim=hidden_dim)
        self.marginal_flow = marginal_flow
        self.gauss_nll = nn.GaussianNLLLoss(full=True, reduction="none")

        self.use_copula = False
        self.clamp_marginals = 1e9
        self.use_gm_param = True
        if len(channel_order) > 0:
            self.channel_order = channel_order
        else:
            self.channel_order = list(range(n_channels))

    def compute_leaf_jll(self, y, xq, mq, context=None):
        channel_indices = xq[:, :, 1].to(int)  # BS x Q
        # Create masks for each channel
        channel_masks = channel_indices.unsqueeze(-1) == torch.arange(self.n_channels, device=xq.device)  # BS x Q x n_channels
        # Apply masks to select relevant data for each channel
        leaf_mask_stacked = mq.unsqueeze(-1).unsqueeze(-1) * channel_masks.unsqueeze(-2)  # BS x Q x 1 x n_channels
        mask_pm = leaf_mask_stacked.permute(0, 3, 2, 1).repeat(1, 1, self.components, 1)  # BS x C x K x Q
        # context \in BS x Q x D x K
        context_stacked = context.unsqueeze(-1) * channel_masks.unsqueeze(-2).unsqueeze(-2)  # BS x Q x D x K x C
        context_stacked = context_stacked.permute(0, 4, 3, 2, 1)  # BS x C x K x D x Q
        context_stacked_compr = compress_masked_tensor(x=context_stacked, mask=mask_pm.unsqueeze(-2).expand_as(context_stacked))  # # BS x C x K x D x Q_
        context_reshaped = context_stacked_compr.transpose(-1, -2).flatten(end_dim=1)  # BS*C x K x Q_ x D
        mask_reshaped = compress_masked_tensor(x=mask_pm, mask=mask_pm).flatten(end_dim=1)  # BS*C* x K x Q_
        y_stacked_compressed = compress_masked_tensor(x=(y.unsqueeze(-1) * channel_masks).transpose(-1, -2), mask=(mq.unsqueeze(-1) * channel_masks).transpose(-1, -2))  # BS x C x Q_
        y_reshaped = y_stacked_compressed.unsqueeze(-2).repeat(1, 1, self.components, 1).flatten(end_dim=1)  # BS*C x K x Q_
        not_empty_mask = mask_reshaped[:, 0, :].sum(-1) > 0
        jll = torch.zeros_like(y_reshaped[:, :, 0])
        jll_non_empty = self.leaf_jll_helper(y=y_reshaped[not_empty_mask], context=context_reshaped[not_empty_mask], mask=mask_reshaped[not_empty_mask])  # BS*C x K
        jll[not_empty_mask] = jll_non_empty
        jll_reshaped = jll.unsqueeze(1).reshape(y.shape[0], self.n_channels, self.components)  # BS x C x K
        return jll_reshaped.transpose(-1, -2)  # BS x K x C

    def standard_gaussian_mnll(self, z: Tensor) -> Tensor:
        r"""Compute NLL of latent distribution, Gaussian"""

        mean = torch.zeros_like(z)

        var = torch.ones_like(z)

        nll = self.gauss_nll(mean, z, var)

        return nll

    def mvn_with_sep_flows_nll(self, z, mean, cov, ldj_of_flow, mask):
        # z : BS x Q
        # ldj: BS
        # mask: BS x Q
        z = z - mean
        ldj = ldj_of_flow.sum(-1)
        l = torch.linalg.cholesky(cov)
        z = torch.linalg.solve_triangular(l, z[:, :, :, None], upper=False)
        z = z[:, :, :, 0].clone()
        # pdb.set_trace()
        log_det_l = torch.log(torch.diagonal(l, dim1=-2, dim2=-1))
        combined_log_det = ldj - log_det_l.sum(-1)  # sub the det of l instead of add the det of l^-1
        z *= mask
        mgnll = self.standard_gaussian_mnll(z) * mask
        gnl = mgnll.sum(-1) - combined_log_det
        njll = gnl * (mask.sum(-1) > 0)
        jll = njll * -1
        return jll

    def uniform_to_gaussian(self, u):
        """Convert uniform [0,1] samples to Gaussian via inverse CDF."""
        # Clamp to avoid numerical issues at boundaries
        normal = torch.distributions.Normal(0, 1)
        return normal.icdf(u)

    def gaussian_to_uniform(self, z):
        """Convert Gaussian samples to uniform [0,1] via CDF."""
        normal = torch.distributions.Normal(0, 1)
        return normal.cdf(z)

    def copula_density(self, R, u, mask):
        """Compute log density handling variable sequence lengths (vectorized)."""
        N = u.shape[-1]
        device = u.device

        original_u = u.clone()
        u = torch.clamp(u, 1e-7, 1 - 1e-7)
        was_clamped = ((original_u != u) * mask).any(dim=-1)
        z = self.uniform_to_gaussian(u)
        # Ensure z has zeros at padded positions
        z = z * mask
        # Compute log determinant for each batch
        sign, logdet = torch.linalg.slogdet(R)  # (B,)

        # Invert R in batch
        R_inv = torch.linalg.inv(R)  # (B, N, N)

        # Compute diff = R_inv - I
        I = torch.ones_like(u).diag_embed(dim1=-2, dim2=-1)
        diff = R_inv - I  # (B, N, N)

        # Quadratic form: z^T * diff * z for each batch
        # quad_form = torch.einsum("bi,bij,bj->b", z, diff, z)  # (B,)
        quad_form = torch.matmul(torch.matmul(z.unsqueeze(-2), diff), z.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        # Copula log density
        log_c = -0.5 * logdet - 0.5 * quad_form  # (B,)
        log_c[was_clamped] = log_c[was_clamped].clamp(max=0)

        return log_c

    def marginal_transform(self, y, mask, context):
        ldj = torch.zeros_like(y)
        if self.marginal_flow == "dsf":
            z = y.clone()
            z, ldj = self.DSF.forward_logdet(context=context, x=z)
            ldj = ldj * mask
            z = z * mask
        if self.marginal_flow == "none":
            z = y
            ldj = torch.zeros_like(y)
        return z, ldj

    def leaf_jll_helper(self, y, context, mask):
        # y in BS*C x K x Q , context in BS*C x K x Q x D, mask in BS*C x K x Q
        if self.use_copula:
            corr = self.gm_param.corr(embeddings=context, mask=mask)
            if self.base_distribution == "normal":
                if self.use_gm_param:
                    mean, std = self.gm_param.gaussian_marginals(context, mask)
                else:
                    mean, std = torch.zeros_like(mask), torch.ones_like(mask)
                # if self.marginal_flow != "none":
                #     std = torch.ones_like(mask)
                z, logdetjacob_flow = self.marginal_transform(y=y, mask=mask, context=context)
                cov = std.unsqueeze(-1) * std.unsqueeze(-2) * corr
                return self.mvn_with_sep_flows_nll(z=z, mean=mean, cov=cov, ldj_of_flow=logdetjacob_flow, mask=mask)
            elif self.base_distribution == "uniform":
                u, logdetjacob_flow = self.marginal_transform(y=y, mask=mask, context=context)
                mll = logdetjacob_flow
                mll = mll.clamp(max=self.clamp_marginals)
                jll = mll.sum(-1) + self.copula_density(u=u, R=corr, mask=mask)
        else:
            mll = self.leaf_mll_helper(y=y, context=context, mask=mask)
            mll = mll.clamp(max=self.clamp_marginals)
            jll = (mll * mask).sum(-1)
        return jll

    def leaf_mll_helper(self, y, context, mask):
        z, logdetjacob_flow = self.marginal_transform(y=y, mask=mask, context=context)
        if self.base_distribution == "uniform":
            return logdetjacob_flow
        elif self.base_distribution == "normal":
            #
            if self.use_gm_param:
                mean, stds = self.gm_param.gaussian_marginals(context, mask)
            else:
                mean, stds = torch.zeros_like(mask), torch.ones_like(mask)
            # if self.marginal_flow != "none":
            #     # this might be wrong.. double check at some point... var = std²?
            #     stds = torch.ones_like(y)
            var = stds**2
            gauss_mnll = self.gauss_nll(z - mean, torch.zeros_like(mean), var)
            mnll = gauss_mnll - logdetjacob_flow
            mll = mnll * -1
            return mll

    def activate_copula(self):
        self.use_copula = True
        print("Activated Correlation matrix network")

    def circuit(self, leaf_loglikl, sum_weights, final_weights):
        return parallel_circuit(leaf_loglikl=leaf_loglikl, sum_weights=sum_weights, final_weights=final_weights)

    def jnll(
        self,
        obs: Tensor,
        mobs: Tensor,
        xq: Tensor,
        mq: Tensor,
        y: Tensor,
    ) -> Tensor:
        """
        Perform MNF forward or inverse pass.

        Args:
            obs: Observed data tensor.
            mobs: Marginal observed data tensor.
            xq: Query data tensor.
            mq: Marginal query data tensor.
            y: Target tensor.
            inverse: Whether to run inverse flow.

        Returns:
            Tuple of tensors (output, log_det, extra).
        """
        # weights are log(weights)
        context, sum_weights, final_weights = self.encoder(obs=obs, mobs=mobs, xq=xq, mq=mq)
        context = torch.stack(context.chunk(chunks=self.components, dim=-1), -1)
        # Parallelize the first loop using advanced indexing
        # Sum over sequence dimension for each channel
        channel_joint = self.compute_leaf_jll(y=y, xq=xq, mq=mq, context=context)  # BS x K x n_channels

        channel_joint = channel_joint[..., self.channel_order]

        # Parallelize the second loop using tensor operations
        jll = self.circuit(leaf_loglikl=channel_joint, sum_weights=sum_weights, final_weights=final_weights)

        return -jll

    def mnll(
        self,
        obs: Tensor,
        mobs: Tensor,
        xq: Tensor,
        mq: Tensor,
        y: Tensor,
    ) -> Tensor:
        """
        Perform MNF forward or inverse pass.

        Args:
            obs: Observed data tensor.
            mobs: Marginal observed data tensor.
            xq: Query data tensor.
            mq: Marginal query data tensor.
            y: Target tensor.
            inverse: Whether to run inverse flow.

        Returns:
            Tuple of tensors (output, log_det, extra).
        """
        bs, q = y.shape
        # weights are log(weights)
        context, sum_weights, final_weights = self.encoder(obs=obs, mobs=mobs, xq=xq, mq=mq)
        context = torch.stack(context.chunk(chunks=self.components, dim=-1), 1)  # shape: BS x K x Q     (K= self.components)
        channel_indices = xq[:, :, 1].to(int)  # BS x Q
        marginal_logliklyhoods = self.leaf_mll_helper(y=y.unsqueeze(1).repeat(1, self.components, 1), context=context, mask=mq.unsqueeze(1).repeat(1, self.components, 1))
        marginal_logliklyhoods = marginal_logliklyhoods.transpose(-1, -2)
        # Create masks for each channel
        channel_masks = channel_indices.unsqueeze(-1) == torch.arange(self.n_channels, device=xq.device)  # BS x Q x n_channels
        # Apply masks to select relevant data for each channel
        leafs_stacked = marginal_logliklyhoods.unsqueeze(-1) * channel_masks.unsqueeze(-2)  # BS x Q x K x n_channels
        leafs_stacked = leafs_stacked[..., self.channel_order]
        mll = self.circuit(
            leaf_loglikl=leafs_stacked.flatten(start_dim=0, end_dim=1),
            sum_weights=sum_weights.repeat(q, 1, 1, 1),
            final_weights=final_weights.repeat(q, 1),
        )

        mll = mll.reshape(bs, q)
        return -mll

    def sample_mvn(self, cov: Tensor, mask: Tensor, n_samples: int):
        bsc, K, Q = mask.shape
        samples = torch.zeros_like(cov[:, :, :, 0]).unsqueeze(-1).repeat(1, 1, 1, n_samples)
        for i in range(bsc):
            real_Q = mask[i, 0].sum(-1)
            rel_mask_cov = mask[i].unsqueeze(-1) & mask[i].unsqueeze(-2)
            cov_masked = cov[i][rel_mask_cov].view(K, real_Q, real_Q)
            mvn = MultivariateNormal(torch.zeros_like(cov_masked[:, :, 0]), covariance_matrix=cov_masked)
            this_samples = mvn.sample(torch.Size([n_samples])).permute(1, 2, 0)
            samples[i][mask[i], :] = this_samples.flatten(end_dim=1)
        return samples

    def sample_leaf_helper(self, context: Tensor, mask: Tensor, n_samples: int):
        # context in BS*C x K x Q x D, mask in BS*C x K x Q
        # TOOD think about sampling. Should all components have the same u ??
        bsc, K, Q = mask.shape
        relevant_mask = mask[:, 0, :].sum(-1) > 0  # BS*C
        bsc_relevant = relevant_mask.sum()
        relevant_masked_mask = mask[relevant_mask].to(bool)
        result = torch.zeros_like(mask).unsqueeze(-1).repeat(1, 1, 1, n_samples)

        # if self.marginal_flow != "none":

        if self.use_gm_param:
            mean, stds = torch.zeros_like((mask[relevant_mask])), torch.ones_like(mask[relevant_mask])
            # mean, stds = self.gm_param.gaussian_marginals(embeddings=context[relevant_mask], mask=mask[relevant_mask])
        else:
            mean, stds = self.gm_param.gaussian_marginals(embeddings=context[relevant_mask], mask=mask[relevant_mask])
        # if self.marginal_flow != "none":
        #     stds = torch.ones_like(mask[relevant_mask])
        if self.use_copula:
            corr = self.gm_param.corr(embeddings=context[relevant_mask], mask=mask[relevant_mask])
            cov = stds.unsqueeze(-1) * stds.unsqueeze(-2) * corr
        else:
            cov = stds.diag_embed(dim1=-2, dim2=-1) ** 2
        # samples_base = torch.cat([mvn.sample(torch.Size([chunk])) for chunk in chunks], axis=0)  # n_samples x bsc x K x Q
        samples_base = self.sample_mvn(cov=cov, mask=relevant_masked_mask, n_samples=n_samples)
        if self.base_distribution == "uniform":
            samples_base = self.gaussian_to_uniform(samples_base).clamp(min=0.05, max=0.95)
        else:
            samples_base += mean.unsqueeze(-1)
        # z = samples_base.permute(1, 2, 3, 0)
        z = samples_base
        samples = torch.zeros_like(z)
        if self.marginal_flow == "dsf":
            z = z[relevant_masked_mask]
            z = self.DSF.inverse(context=context[relevant_mask][relevant_masked_mask], u=z)
            samples[relevant_masked_mask] = z
        if self.marginal_flow == "none":
            samples = z
        result[relevant_mask] = samples
        return result

    def sample_leaves(self, channel_indeces: Tensor, mq: Tensor, n_samples: int, context: Tensor = None):
        # Create masks for each channel
        channel_masks = channel_indeces.unsqueeze(-1) == torch.arange(self.n_channels, device=mq.device)  # BS x Q x n_channels
        # Apply masks to select relevant data for each channel
        leaf_mask_stacked = mq.unsqueeze(-1).unsqueeze(-1) * channel_masks.unsqueeze(-2)  # BS x Q x 1 x n_channels
        mask_pm = leaf_mask_stacked.permute(0, 3, 2, 1).repeat(1, 1, self.components, 1)  # BS x C x K x Q
        # context \in BS x Q x D x K
        context_stacked = context.unsqueeze(-1) * channel_masks.unsqueeze(-2).unsqueeze(-2)  # BS x Q x D x K x C
        context_stacked = context_stacked.permute(0, 4, 3, 2, 1)  # BS x C x K x D x Q
        # context_stacked_compr = compress_masked_tensor(x=context_stacked, mask=mask_pm.unsqueeze(-2).expand_as(context_stacked))  # # BS x C x K x D x Q_
        context_reshaped = context_stacked.transpose(-1, -2).flatten(end_dim=1)  # BS*C x K x Q_ x D
        samples = self.sample_leaf_helper(context=context_reshaped, mask=mask_pm.flatten(end_dim=1), n_samples=n_samples)  # BS*C x K x Q x n_samples
        samples_reshaped = samples.unsqueeze(1).reshape(mq.shape[0], self.n_channels, self.components, -1, n_samples)  # BS x C x K x Q x n_samples
        return samples_reshaped.permute(0, 3, 4, 2, 1)  # BS x Q x n_samples x K x C

    def sample(self, obs: Tensor, mobs: Tensor, xq: Tensor, mq: Tensor, n_samples: int = 10) -> Tensor:
        # weights are log(weights)
        context, sum_weights, final_weights = self.encoder(obs=obs, mobs=mobs, xq=xq, mq=mq)
        context = torch.stack(context.chunk(chunks=self.components, dim=-1), -1)
        bs, q = mq.shape

        # Pre-calculate indices for gathering
        channel_indices = xq[:, :, 1].to(torch.long)

        # 1. Get all leaf samples for all layers at once
        # Shape: [BS, Q, n_samples, K, Channels] (Assuming sample_leaves returns this structure)
        leafs_stacked = self.sample_leaves(context=context, n_samples=n_samples, channel_indeces=channel_indices, mq=mq)
        leafs_stacked = leafs_stacked[..., self.channel_order]

        # 2. Start Sampling from the ROOT (The end of the circuit)
        # final_weights determines the active components at the highest layer
        final_probs = torch.exp(final_weights)  # [BS, K]

        # [BS, n_samples]
        current_node_indices = torch.multinomial(final_probs, num_samples=n_samples, replacement=True)

        samples = torch.zeros(self.n_channels, bs, q, n_samples).to(mq.device)

        # Helper to gather leaf samples: [BS, Q, n_samples, K] -> [BS, Q, n_samples]

        def gather_leaf_samples(layer_idx, leaf_indices):
            # layer_leaves: [BS, Q, n_samples, K]
            layer_leaves = leafs_stacked[..., layer_idx]
            # Expand indices to [BS, Q, n_samples, 1]
            gather_idx = leaf_indices.unsqueeze(1).unsqueeze(-1).expand(-1, q, -1, -1)
            return torch.gather(layer_leaves, -1, gather_idx).squeeze(-1)

        for i in reversed(range(1, self.n_channels)):
            # weights: [BS, K^2, K]
            # (This represents Log-Probs of Source(K^2) -> Target(K))
            weights = sum_weights[:, i - 1]

            # We need to pick the specific column 'k' (Target) for each sample.
            # But we must keep all 'K^2' rows (Sources) to sample from them.

            # 1. Expand weights to accommodate n_samples dimension
            # Input: [BS, K^2, K] -> [BS, K^2, 1, K] -> [BS, K^2, n_samples, K]
            w_expanded = weights.unsqueeze(2).expand(-1, -1, n_samples, -1)

            # 2. Prepare indices for gathering
            # current_node_indices: [BS, n_samples] -> [BS, 1, n_samples, 1]
            # We expand to [BS, K^2, n_samples, 1] to "pick the same target column" for every source row.
            gather_idx = current_node_indices.unsqueeze(1).unsqueeze(-1).expand(-1, self.components**2, -1, -1)

            # 3. Gather along the last dimension (The Target Node K)
            # We get the log-probabilities of all K^2 sources that could lead to our current node K
            # Result: [BS, K^2, n_samples, 1] -> Squeeze to [BS, K^2, n_samples]
            selected_weights = torch.gather(w_expanded, -1, gather_idx).squeeze(-1)

            # 4. Prepare for Sampling
            # We need to sample over the K^2 dimension (dim 1).
            # Permute to [BS, n_samples, K^2] for standard softmax/multinomial handling
            logits = selected_weights.permute(0, 2, 1)

            # 5. Normalize and Sample
            # Softmax over the K^2 possible source pairs
            probs = torch.softmax(logits, dim=-1)

            # Flatten for multinomial: [BS * n_samples, K^2]
            probs_flat = probs.reshape(-1, self.components**2)

            # Sample the source index
            # source_indices_flat: [BS * n_samples, 1]
            source_indices_flat = torch.multinomial(probs_flat, 1)
            source_indices = source_indices_flat.view(bs, n_samples)

            # 6. Decode the Source Index (prev_node * K + leaf)
            prev_node_indices = torch.div(source_indices, self.components, rounding_mode="floor")
            leaf_indices = source_indices % self.components

            samples[i] = gather_leaf_samples(i, leaf_indices)
            current_node_indices = prev_node_indices

        # 7. Handle the Base Case (Layer 0)
        # The loop ends, leaving us with indices for Layer 0
        samples[0] = gather_leaf_samples(0, current_node_indices)

        # 8. Final Formatting
        # samples: [Channels, BS, Q, n_samples] -> [BS, Channels, Q, n_samples]
        samples = samples.permute(1, 0, 2, 3)

        # Select specific channel per query if required
        gather_idx = channel_indices.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, n_samples)
        final_samples = torch.gather(samples, 1, gather_idx).squeeze(1)

        return final_samples
