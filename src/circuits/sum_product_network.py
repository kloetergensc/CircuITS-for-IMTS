import math

import torch


def circuit(leaf_loglikl, sum_weights, final_weights):
    bs, k, n_channels = leaf_loglikl.shape
    node_ll = leaf_loglikl[:, :, 0]  # BS x K
    for i in range(1, n_channels):
        cp = leaf_loglikl[:, :, i]  # BS x K
        product_nodes = (node_ll.unsqueeze(-1) + cp.unsqueeze(-2)).flatten(start_dim=-2)
        node_ll = torch.logsumexp((product_nodes.unsqueeze(-1) + sum_weights[:, i - 1]), 1)
    return torch.logsumexp((node_ll + final_weights), -1)


def parallel_log_matmul_scan(log_matrices):
    """Computes the cumulative product of a sequence of log-matrices (Log-Semiring Scan)."""

    def log_matmul(a, b):
        # Log-Space Matrix Mul: C_ij = LSE_k(A_ik + B_kj)
        # Dimensions: a (..., K, K), b (..., K, K) -> Result (..., K, K)
        # LSE over the shared dimension (K), which is dimension -2 after unsqueezing/broadcasting
        return torch.logsumexp(a.unsqueeze(-1) + b.unsqueeze(-3), dim=-2)

    n_steps = log_matrices.shape[1]
    current_val = log_matrices

    # Iterate log2(N) times (Hillis-Steele algorithm)
    for i in range(int(math.ceil(math.log2(n_steps)))):
        shift = 2**i
        if shift >= n_steps:
            break

        # Shift and multiply
        left = current_val[:, :-shift, :, :]
        right = current_val[:, shift:, :, :]

        updated_right = log_matmul(left, right)

        # Concatenate: The first 'shift' elements don't change
        current_val = torch.cat([current_val[:, :shift], updated_right], dim=1)

    return current_val


def parallel_circuit(leaf_loglikl, sum_weights, final_weights):
    """
    Parallel implementation of the SPN forward pass (Log-Semiring Recurrence).

    leaf_loglikl: BS x K x C
    sum_weights:  BS x (C-1) x K^2 x K
    final_weights:BS x K
    """
    bs, k, n_channels = leaf_loglikl.shape
    n_sequence_steps = n_channels - 1

    # --- STEP 1: Construct Transition Matrices (M_t) in Parallel ---
    # Goal: Calculate M_t[j, m] = LSE_k( leaf_t[k] + W_t[j, k, m] ) for all t=1 to C-1

    # 1a. Isolate the Leaf Log-Likelihoods for all channels t=1 to C-1
    # Shape: BS x (C-1) x K (Transpose to put channels first after BS)
    cp_all = leaf_loglikl[:, :, 1:].transpose(1, 2)

    # 1b. Reshape sum_weights to expose (Prev_Node, Leaf_Node, Next_Node)
    # Shape: BS x (C-1) x K (Prev) x K (Leaf) x K (Next)
    weights_reshaped = sum_weights.view(bs, n_sequence_steps, k, k, k)

    # 1c. Prepare cp_all for broadcasting:
    # Shape: BS x (C-1) x 1 x K x 1 (1 for Prev, 1 for Next)
    cp_broadcast = cp_all.unsqueeze(2).unsqueeze(4)

    # 1d. Compute the Transition Matrices (M_t):
    # Sum over the Leaf_Node dimension 'k' (index 3).
    # transitions shape: BS x (C-1) x K (Prev) x K (Next)
    transitions = torch.logsumexp(cp_broadcast + weights_reshaped, dim=3)

    # --- STEP 2: Parallel Scan (Accumulate Products) ---
    # Compute M_total = M_1 * M_2 * ... * M_(C-1)
    # The scan returns all prefix products: M_1, M_1*M_2, M_1*M_2*M_3, ...
    cum_transitions = parallel_log_matmul_scan(transitions)

    # --- STEP 3: Initial State & Final Reduction (Matches Original Code) ---

    # 3a. Initial state from the first channel (t=0)
    # node_ll_0 shape: BS x K
    node_ll_0 = leaf_loglikl[:, :, 0]

    # 3b. Use the LAST accumulated product (M_total)
    # Total matrix shape: BS x K x K
    total_matrix = cum_transitions[:, -1, :, :]

    # 3c. Final vector-matrix multiplication (L_final = L_0 @ M_total)
    # L_final[m] = LSE_j( L_0[j] + M_total[j, m] )
    # node_ll shape: BS x K (The K sum nodes from the last layer)
    node_ll = torch.logsumexp(node_ll_0.unsqueeze(-1) + total_matrix, dim=1)

    # 3d. Final root reduction (Matches the last line of your original code)
    return torch.logsumexp(node_ll + final_weights, -1)
