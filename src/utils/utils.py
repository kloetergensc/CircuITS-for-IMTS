import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import simpson


def test_model_integrates_to_one(model, bounds=(-10, 10)):
    """
    Test that the model's joint likelihood integrates to 1.

    Args:
        model: The model with a joint likelihood method
        bounds: Integration bounds as tuple (min, max)
    """

    # Convert to tensor and get likelihood

    # Perform double integration
    # Monte Carlo integration
    number_samples = int(1e7)
    a_samples = torch.rand(number_samples) * (bounds[1] - bounds[0]) + bounds[0]
    b_samples = torch.rand(number_samples) * (bounds[1] - bounds[0]) + bounds[0]
    # c_samples = torch.rand(number_samples) * (bounds[1] - bounds[0]) + bounds[0]

    area = (bounds[1] - bounds[0]) ** 2

    obs = torch.tensor(
        [
            [
                [0.0000, 0.0000, 0.1656],
                [0.0000, 0.0000, 0.5188],
                [0.0000, 0.0000, 0.9515],
                [0.0000, 0.0000, 0.1612],
                [0.1000, 0.0000, 0.9836],
                [0.1000, 0.0000, 0.7495],
                [0.1000, 0.0000, 0.6935],
                [0.1000, 0.0000, 0.5803],
                [0.2000, 0.0000, 0.2822],
                [0.2000, 0.0000, 0.2230],
            ]
        ]
    ).to(model.device)
    mobs = torch.ones_like(obs[:, :, 0]).to(model.device)
    xq = torch.tensor(
        [
            [
                [0.8, 1],
                [0.9, 1],
                # [0.9, 0],
                [0.85, 0],
                [0.85, 1],
            ]
        ]
    ).to(model.device)
    mq = torch.cat(
        [
            torch.ones(1, 2),
            torch.zeros(1, 2),
        ],
        axis=-1,
    ).to(model.device)
    y = torch.stack(
        [
            a_samples,
            b_samples,
            # c_samples,
            torch.zeros_like(a_samples),
            torch.zeros_like(a_samples),
        ],
        axis=-1,
    ).to(model.device)
    bs = 10000
    y_chunked = y.chunk(number_samples // bs, axis=0)
    likelihood_sum = 0
    for y in y_chunked:
        with torch.no_grad():
            nlog_likelihood = model.jnll(obs=obs.repeat(bs, 1, 1), mobs=mobs.repeat(bs, 1), xq=xq.repeat(bs, 1, 1), mq=mq.repeat(bs, 1), y=y)
            likelihood_sum += (torch.exp(-nlog_likelihood)).sum().item()

    integral_result = (area * likelihood_sum) / number_samples

    print(f"Integral: {integral_result:.6f}")

    return integral_result


def test_sample_likelihood_consistency(model, bounds=(-5, 5), n_samples=100000, n_query_samples=100000):
    """
    Test that samples from the model align with the predicted likelihoods using Monte Carlo.

    Args:
        model: The model with sample and jnll methods
        bounds: Integration bounds as tuple (min, max)
        n_samples: Number of Monte Carlo samples for likelihood estimation
        n_query_samples: Number of samples to generate from the model
    """
    # Fixed conditioning data
    obs = torch.tensor(
        [
            [
                [0.0000, 0.0000, 0.1656],
                [0.0000, 0.0000, 0.5188],
                [0.0000, 0.0000, 0.9515],
                [0.0000, 0.0000, 0.1612],
                [0.1000, 0.0000, 0.9836],
                [0.1000, 0.0000, 0.7495],
                [0.1000, 0.0000, 0.6935],
                [0.1000, 0.0000, 0.5803],
                [0.2000, 0.0000, 0.2822],
                [0.2000, 0.0000, 0.2230],
            ]
        ]
    ).to(model.device)
    mobs = torch.ones_like(obs[:, :, 0]).to(model.device)
    xq = torch.tensor(
        [
            [
                [0.8, 0],
                # [0.9, 1],
                [0.85, 3],
            ]
        ]
    ).to(model.device)
    mq = torch.ones_like(xq[:, :, 0]).to(model.device)

    # Generate samples from the model
    with torch.no_grad():
        model_samples = model.sample(obs=obs, mobs=mobs, xq=xq, mq=mq, n_samples=n_query_samples)
        # Shape: [batch_size, n_query_points, n_samples]
        model_samples = model_samples[0].T  # Shape: [n_samples, n_query_points]

    # Create a grid of test points
    a_test = torch.rand(n_samples) * (bounds[1] - bounds[0]) + bounds[0]
    b_test = torch.rand(n_samples) * (bounds[1] - bounds[0]) + bounds[0]
    c_test = torch.rand(n_samples) * (bounds[1] - bounds[0]) + bounds[0]

    test_points = torch.stack(
        [
            a_test,
            b_test,
            # c_test,
        ],
        dim=-1,
    ).to(model.device)

    # Compute likelihood at test points
    bs = 100000
    test_chunked = test_points.chunk(n_samples // bs, dim=0)
    log_likelihoods = []

    for test_chunk in test_chunked:
        chunk_size = test_chunk.shape[0]
        with torch.no_grad():
            nlog_likelihood = model.jnll(obs=obs.repeat(chunk_size, 1, 1), mobs=mobs.repeat(chunk_size, 1), xq=xq.repeat(chunk_size, 1, 1), mq=mq.repeat(chunk_size, 1), y=test_chunk)
            log_likelihoods.append(-nlog_likelihood)

    log_likelihoods = torch.cat(log_likelihoods, dim=0)
    likelihoods = torch.exp(log_likelihoods)

    # Estimate empirical density from model samples using kernel density estimation
    bandwidth = 0.01
    empirical_densities = torch.zeros(n_samples).to(model.device)

    for i, test_point in enumerate(test_points):
        # Compute distance from test point to all model samples
        distances = torch.norm(model_samples - test_point.unsqueeze(0), dim=1)
        # Gaussian kernel
        kernel_values = torch.exp(-0.5 * (distances / bandwidth) ** 2)
        empirical_densities[i] = kernel_values.mean() / (bandwidth**2 * 2 * torch.pi)

    # Normalize empirical densities to integrate to 1 (approximate)
    area = (bounds[1] - bounds[0]) ** 2
    empirical_densities = empirical_densities / (empirical_densities.mean() * area)

    # Compare likelihood and empirical density
    # likelihood_np = likelihoods.cpu().numpy()
    # empirical_np = empirical_densities.cpu().numpy()

    # Compute correlation
    correlation = torch.corrcoef(torch.stack([likelihoods, empirical_densities]))[0, 1].item()

    # Compute KL divergence (approximate)
    epsilon = 1e-8
    empirical_norm = empirical_densities / (empirical_densities.sum() + epsilon)
    likelihood_norm = likelihoods / (likelihoods.sum() + epsilon)
    kl_div = torch.sum(likelihood_norm * torch.log((likelihood_norm + epsilon) / (empirical_norm + epsilon))).item()

    print(f"Correlation between likelihood and empirical density: {correlation:.6f}")
    print(f"KL divergence: {kl_div:.6f}")
    print(f"Mean likelihood: {likelihoods.mean().item():.6f}")
    print(f"Mean empirical density: {empirical_densities.mean().item():.6f}")

    return {"correlation": correlation, "kl_divergence": kl_div, "mean_likelihood": likelihoods.mean().item(), "mean_empirical_density": empirical_densities.mean().item()}


# def test_sample_likelihood_consistency(model, loader, n_samples=100):
#    mse_sum = 0
#    n_instances = 0
#    for batch in loader:
#        obs, mobs, xq, mq, y = (tensor.to(model.device) for tensor in batch)
#        samples = model.sample(obs=obs, mobs=mobs, xq=xq, mq=mq, n_samples=n_samples)
#        yhat_stack = samples.permute(2, 0, 1)
#        jl_samples_list = []
#        for yhat in yhat_stack:
#            jnll_of_samples = model.jnll(obs=obs, mobs=mobs, xq=xq, mq=mq, y=yhat)
#            jl_samples_list.append(torch.exp(-jnll_of_samples))
#        jl_samples = torch.stack(jl_samples_list, -1)
#        jl_samples_mean = jl_samples.mean(-1)
#        njnll_target = model.jnll(obs=obs, mobs=mobs, xq=xq, mq=mq, y=y)
#        jl_target = torch.exp(-njnll_target)
#        mse_sum += torch.sum((jl_target - jl_samples_mean) ** 2).item()
#        n_instances += y.shape[0]
#    print("joined sampled vs target-liklyhood", (mse_sum / n_instances))
#


def evaluate_njnll(model, loader):
    njnll_sum = 0.0
    count = 0.0
    for batch in loader:
        obs, mobs, xq, mq, y = (tensor.to(model.device) for tensor in batch)

        if xq.shape[1] == 0:
            continue
        jnll = model.jnll(obs=obs, mobs=mobs, xq=xq, mq=mq, y=y)
        njnll = jnll / (mq.sum(1) + (mq.sum(1) < 1).to(int))
        nsamples = (mq.sum(-1) > 0).sum()
        njnll_sum += (njnll[mq.sum(1) > 0]).sum()
        count += nsamples
    mean_njnll = njnll_sum / count
    return mean_njnll


def evaluate_mnll(model, loader):
    mnll_sum = 0.0
    count = 0
    for batch in loader:
        obs, mobs, xq, mq, y = (tensor.to(model.device) for tensor in batch)
        if xq.shape[1] == 0:
            continue
        mnll = model.mnll(obs=obs, mobs=mobs, xq=xq, mq=mq, y=y)
        mnll_sum += (mnll[mq.to(bool)]).sum()
        count += mq.sum()
    mean_mnll = mnll_sum / count
    return mean_mnll


def compress_masked_tensor(x, mask):
    """
    Compress tensor (A,B,C,D) along D using boolean mask (A,B,1,D),
    keeping valid elements and zero-padding to the maximum valid count (D*).
    """
    mask = mask.to(bool)
    lengths = mask.sum(dim=-1)  # (A,B,C)
    compr_lengths = lengths.max().item()  # resulting width

    # Compute gather indices
    idx = torch.argsort((~mask).to(int), dim=-1)  # valid first
    idx = idx[..., :compr_lengths]  # truncate compr_lengths

    # Gather valid elements
    compressed = torch.gather(x, dim=-1, index=idx)

    # Zero out invalid beyond length per (A,B,C)
    return compressed


def get_gpu_memory_allocated():
    # Memory explicitly used by tensors
    allocated = torch.cuda.memory_allocated() / 1024**3
    # Memory reserved by caching allocator (the "blob" PyTorch holds onto)
    reserved = torch.cuda.memory_reserved() / 1024**3

    return allocated


def sample_based_metrics(model, loader):
    crps_sum = 0
    energy_score_sum = 0
    mse_sum = 0
    sum_queries = 0
    sum_instances = 0
    for batch in loader:
        obs, mobs, xq, mq, y = (tensor.to(model.device) for tensor in batch)
        if mq.sum() == 0:
            continue
        samples = model.sample(obs=obs, mobs=mobs, xq=xq, mq=mq, n_samples=1000)
        for i in range(y.shape[0]):
            n_queries = mq[i].sum().item()
            if n_queries > 0:
                crps = compute_crps(yhat=samples[i], y=y[i], mask=mq[i])
                crps_sum += crps * n_queries
                energy_score_sum += compute_energy_score(yhat=samples[i : i + 1], y=y[i : i + 1], mask=mq[i : i + 1])
                mse_sum += compute_mse(yhat=samples[i], y=y[i], mask=mq[i]) * n_queries
                sum_queries += n_queries
                sum_instances += 1
    return {
        "mse": mse_sum / sum_queries,
        "crps": crps_sum / sum_queries,
        "energy_score": energy_score_sum / sum_instances,
    }


def compute_energy_score(yhat, y, mask, beta=1):
    """
    y: [batch, dim]
    yhat: [batch, dim, nsamples]
    mask: optional, [batch, dim]
    """

    nsamples = yhat.shape[-1]

    yhat = yhat.permute(0, 2, 1)

    valid_batch = mask.sum(dim=1) > 0
    y = y[valid_batch] * mask[valid_batch]
    yhat = yhat[valid_batch] * mask[valid_batch, None, :]

    # First term: mean over samples
    diff = torch.cdist(yhat, y[:, None, :], p=2)  # [batch, nsamples, 1]
    first_term = diff.pow(beta).mean(dim=1).squeeze(-1)  # mean over nsamples

    # Second term: pairwise distances, exclude diagonal
    pairwise = torch.cdist(yhat, yhat, p=2)
    diag = torch.diagonal(pairwise, dim1=1, dim2=2)
    pairwise = pairwise - torch.diag_embed(diag)
    second_term = -pairwise.pow(beta).sum(dim=(1, 2)) / (2 * nsamples**2)

    energy = first_term + second_term
    return energy.sum() / y.shape[0]  # per batch


def compute_mse(yhat, y, mask):
    means = torch.mean(yhat, -1)
    squared_error = (y - means) ** 2
    sse = squared_error * mask
    mse = sse.sum() / mask.sum()
    return mse


def compute_crps(yhat, y, mask):
    y_expanded = y.unsqueeze(-1)  # [B, 1, K]
    term1 = torch.mean(torch.abs(y_expanded - yhat), dim=-1)  # [B, K]

    # Compute pairwise differences between ensemble members
    yhat1 = yhat.unsqueeze(-2)  # [B, K, 1, nsamples]
    yhat2 = yhat.unsqueeze(-1)  # [B, K, nsamples, 1]
    term2 = 0.5 * torch.mean(torch.abs(yhat1 - yhat2), dim=(-1, -2))  # [B, K]

    crps = term1 - term2
    crps *= mask
    # Apply mask
    return crps.sum() / mask.sum()


def plot_mll(model, loader, path):
    """
    Plot a histogram of all per-query marginal negative log-likelihoods (mll/mnll)
    collected over the given loader. Displays the histogram and returns the values.
    """
    os.makedirs(path, exist_ok=True)
    mll_list = []
    with torch.no_grad():
        for batch in loader:
            obs, mobs, xq, mq, y = (tensor.to(model.device) for tensor in batch)
            if xq.shape[1] == 0:
                continue
            mnll = model.mnll(obs=obs, mobs=mobs, xq=xq, mq=mq, y=y)
            mask = mq.to(bool)
            if mask.any():
                mll_vals = mnll[mask].cpu().numpy()
                mll_list.append(mll_vals)

    if not mll_list:
        print("No MLL values found in loader.")
        return None

    all_mll = np.concatenate(mll_list, axis=0)

    plt.figure(figsize=(8, 6))
    plt.hist(all_mll, bins=50, color="C0", alpha=0.75)
    plt.xlabel("Marginal negative log-likelihood (mnll)")
    plt.ylabel("Count")
    plt.title("Histogram of per-query MLLs")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{path}/mll.png")

    return all_mll


def plot_crps(model, loader, path, n_samples=1000):
    """
    Plot a histogram of per-query CRPS values collected over the given loader.
    Saves the histogram to path/crps.png and returns the CRPS values as a numpy array.
    """
    os.makedirs(path, exist_ok=True)
    crps_list = []
    with torch.no_grad():
        for batch in loader:
            obs, mobs, xq, mq, y = (tensor.to(model.device) for tensor in batch)
            if xq.shape[1] == 0:
                continue

            samples = model.sample(obs=obs, mobs=mobs, xq=xq, mq=mq, n_samples=n_samples)
            # samples: [B, K, nsamples], y: [B, K], mq: [B, K]
            mask = mq.to(bool)
            if not mask.any():
                continue

            # Compute per-query CRPS without reduction
            y_expanded = y.unsqueeze(-1)  # [B, K, 1]
            term1 = torch.mean(torch.abs(y_expanded - samples), dim=-1)  # [B, K]

            yhat1 = samples.unsqueeze(-2)  # [B, K, 1, nsamples]
            yhat2 = samples.unsqueeze(-1)  # [B, K, nsamples, 1]
            term2 = 0.5 * torch.mean(torch.abs(yhat1 - yhat2), dim=(-1, -2))  # [B, K]

            crps_per_query = term1 - term2
            # Apply mask and collect valid values
            crps_vals = crps_per_query[mask].cpu().numpy()
            if crps_vals.size:
                crps_list.append(crps_vals)

    if not crps_list:
        print("No CRPS values found in loader.")
        return None

    all_crps = np.concatenate(crps_list, axis=0)

    plt.figure(figsize=(8, 6))
    plt.hist(all_crps, bins=50, color="C1", alpha=0.75)
    plt.xlabel("CRPS")
    plt.ylabel("Count")
    plt.title("Histogram of per-query CRPS")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{path}/crps.png", dpi=300, bbox_inches="tight")
    plt.clf()

    return all_crps


def plot_crps_mll(model, loader, path, n_samples=1000):
    """
    Scatter plot of per-query marginal negative log-likelihood (mll) vs CRPS.

    Args:
        model: model with mnll and sample methods
        loader: data loader yielding (obs, mobs, xq, mq, y)
        path: directory to save the plot
        n_samples: number of samples to draw for CRPS estimation
    Returns:
        (all_mll, all_crps) as numpy arrays
    """
    os.makedirs(path, exist_ok=True)
    mll_list = []
    crps_list = []
    with torch.no_grad():
        for batch in loader:
            obs, mobs, xq, mq, y = (tensor.to(model.device) for tensor in batch)
            if xq.shape[1] == 0:
                continue

            mask = mq.to(bool)
            if not mask.any():
                continue

            # marginal negative log-likelihood per query
            mnll = model.mnll(obs=obs, mobs=mobs, xq=xq, mq=mq, y=y)  # [B, K]

            # samples for CRPS: [B, K, nsamples]
            samples = model.sample(obs=obs, mobs=mobs, xq=xq, mq=mq, n_samples=n_samples)

            # compute per-query CRPS (term1 - term2) like in plot_crps
            y_expanded = y.unsqueeze(-1)  # [B, K, 1]
            term1 = torch.mean(torch.abs(y_expanded - samples), dim=-1)  # [B, K]

            yhat1 = samples.unsqueeze(-2)  # [B, K, 1, nsamples]
            yhat2 = samples.unsqueeze(-1)  # [B, K, nsamples, 1]
            term2 = 0.5 * torch.mean(torch.abs(yhat1 - yhat2), dim=(-1, -2))  # [B, K]

            crps_per_query = term1 - term2  # [B, K]

            # collect masked values
            mll_vals = mnll[mask].cpu().numpy()
            crps_vals = crps_per_query[mask].cpu().numpy()

            if mll_vals.size:
                mll_list.append(mll_vals)
                crps_list.append(crps_vals)

    if not mll_list:
        print("No values found for CRPS vs MLL plot.")
        return None

    all_mll = np.concatenate(mll_list, axis=0)
    all_crps = np.concatenate(crps_list, axis=0)

    # scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(all_mll, all_crps, s=10, alpha=0.6)
    plt.xlabel("Marginal negative log-likelihood (mnll)")
    plt.ylabel("CRPS")
    # compute and show correlation in title if possible
    try:
        corr = np.corrcoef(all_mll, all_crps)[0, 1]
        plt.title(f"CRPS vs MLL (corr={corr:.3f})")
    except Exception:
        plt.title("CRPS vs MLL")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{path}/crps_mll.png", dpi=300, bbox_inches="tight")
    plt.clf()

    return all_mll, all_crps
