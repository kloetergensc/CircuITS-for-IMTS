import os

import matplotlib.pyplot as plt
import numpy as np
import torch


# --- Style Helper ---
def publication_style():
    """Sets Matplotlib parameters for ICML-quality figures (Serif, PDF, tight)."""
    return plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 300,
            "axes.linewidth": 0.8,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


FIG_WIDTH = 4 / 2.5
FIG_HEIGT = 1.5 / 2.5
LINEWIDTH = 0.5
ALPHA = 0.33


# --- FUNCTION 1: Plot Ground Truth Examples ---
def plot_dataset_examples(path, dataloader, n_channels, n_examples=5, channel_names=None):
    """
    Iterates over the dataloader to collect and plot N distinct examples
    from the ground truth dataset.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    COLOR_Y = "#1AA1D6"  # Red for Ground Truth

    collected_count = 0
    did_obs = 0

    with publication_style():
        # Iterate until we have enough examples
        fig, axes = plt.subplots(n_channels, 1, figsize=(FIG_WIDTH, FIG_HEIGT * n_channels), sharex=True, constrained_layout=True)
        if n_channels == 1:
            axes = [axes]

        for batch in dataloader:
            if collected_count >= n_examples:
                break

            # Unpack batch (assuming standard structure)
            # We only need the full trajectory (obs + ground truth) for visualization
            # Usually y contains the values to predict, xq the times.
            # Adjust these indices if your loader structure differs.
            obs, mobs, xq, mq, y = batch

            batch_size = y.shape[0]

            for i in range(batch_size):

                if collected_count >= n_examples:
                    break

                # Plot setup

                fig.suptitle("Ground Truth", weight="bold")

                for channel in range(n_channels):
                    ax = axes[channel]
                    if not did_obs:
                        o_mask = torch.logical_and(mobs[i, :].bool(), (obs[i, :, 1] == channel))
                        o_times = obs[i, :, 0][o_mask].cpu().numpy()
                        o_vals = obs[i, :, -1][o_mask].cpu().numpy()
                        ax.plot(o_times, o_vals, color="#808080", alpha=1, linewidth=1)
                    # Extract Ground Truth for this channel
                    mask = torch.logical_and(mq[i, :].bool(), (xq[i, :, 1] == channel))
                    times = xq[i, :, 0][mask].cpu().numpy()
                    values = y[i, :][mask].cpu().numpy()

                    # Sort for plotting
                    sort_idx = np.argsort(times)
                    times = times[sort_idx]
                    values = values[sort_idx]

                    # Plot
                    ax.plot(list(o_times[-1:]) + list(times), list(o_vals[-1:]) + list(values), color=COLOR_Y, linewidth=LINEWIDTH, alpha=ALPHA)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_ylim((-2.5, 2.5))
                    # ax.plot(o_times, o_vals, color=COLOR_OBS, marker="x", markersize=1, linewidth=1.5, alpha=0.5)

                    # Styling
                    # ylab = channel_names[channel] if channel_names and len(channel_names) > channel else f"C-{channel+1}"
                    # ax.set_ylabel(ylab)
                    # ax.grid(True, linestyle="--", alpha=0.4)
                    # if channel == 0:
                    #    ax.legend(loc="upper right", frameon=False)
                    ax.spines["bottom"].set_visible(True)
                    ax.spines["bottom"].set_color("black")
                    ax.spines["left"].set_visible(True)
                    ax.spines["left"].set_color("black")
                    ax.spines["right"].set_visible(True)
                    ax.spines["right"].set_color("black")
                    ax.spines["top"].set_visible(True)
                    ax.spines["top"].set_color("black")

                did_obs = 1
                axes[-1].set_xlabel("Time $t$")
                collected_count += 1
                # Save
        save_file = f"{path}_ground_truth_{collected_count}.pdf"
        fig.savefig(save_file, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved dataset plot: {save_file}")


# --- FUNCTION 2: Plot Single Instance Inference ---
def plot_instance_inference(path, model, batch, instance_idx=0, n_samples=10, channel_names=None, color="#E916D7", model_name="CircuITS"):
    """
    Takes a SINGLE instance from a batch, generates n_samples from the model,
    and plots them against observations and ground truth.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Colors
    # COLOR_GT = "#D62728"  # Red

    # Unpack the specific instance

    # Run Model
    # samples shape: [1, n_time_points, n_samples]
    if "CircuITS" in model_name:
        obs, mobs, xq, mq, y = (tensor.to(model.device) for tensor in batch)
        q_c = xq[:, :, 1]
        q_t = xq[:, :, 0]
        x_c = obs[:, :, 1]
        x_t = obs[:, :, 0]
        x_v = obs[:, :, 2]
        samples = model.sample(obs=obs, mobs=mobs, xq=xq, mq=mq, n_samples=n_samples)
    elif model_name == "ProFITi":
        x_t, x_c, x_v, mobs, q_t, q_c, y, mq = (tensor.to(model.device) for tensor in batch)
        model.distribution(x_t, x_c, x_v, mobs, q_t, q_c, mq)
        samples, _ = model.samples(mq, nsamples=n_samples)
        samples = samples.permute(0, 2, 1)
    elif "MOSES" in model_name:
        obs, mobs, qry, mq, y = batch
        x_t = obs[:, :, 0]
        x_c = obs[:, :, 1]
        x_v = obs[:, :, 2]
        q_t = qry[:, :, 0]
        q_c = qry[:, :, 1]
        model.to(y.device)
        model(x_t, x_c, mobs, x_v, q_t, q_c, mq)
        samples = model.sample(mq=mq, num_samples=n_samples)
    else:
        raise NotADirectoryError("illegal model_name")
    with publication_style():
        n_channels = int(torch.max(x_c).item()) + 1
        fig, axes = plt.subplots(n_channels, 1, figsize=(FIG_WIDTH, FIG_HEIGT * n_channels), sharex=True, constrained_layout=True)
        if n_channels == 1:
            axes = [axes]

        fig.suptitle(model_name, weight="bold")

        for channel in range(n_channels):
            ax = axes[channel]
            ax.set_ylim((-2.5, 2.5))
            # 1. Prepare Data Indices
            # Query/Prediction Mask
            q_mask = torch.logical_and(mq[0, :].bool(), (q_c[0, :] == channel))
            q_times = q_t[0, :][q_mask].cpu().numpy()
            q_sort = np.argsort(q_times)
            q_times = q_times[q_sort]

            # Observation Mask
            for i in range(mq.shape[0]):
                o_mask = torch.logical_and(mobs[i, :].bool(), (x_c[i, :] == channel))
                o_times = x_t[i, :][o_mask].cpu().numpy()
                o_vals = x_v[i, :][o_mask].cpu().numpy()
                ax.plot(o_times, o_vals, color="#808080", alpha=1, linewidth=1)
                break

            # 2. Plot Model Samples
            # samples tensor usually: [batch, time, sample_count]
            # Create color map for samples if rainbow mode is enabled
            if color == "rainbow":
                colors = plt.cm.tab10(np.linspace(0, 1, n_samples))
            else:
                colors = [color] * n_samples
            for s in range(n_samples):
                # Extract specific sample, mask it, sort it
                s_vals = samples[0, :, s][q_mask].cpu().numpy()[q_sort]

                ax.plot(list(o_times[-1:]) + list(q_times), list(o_vals[-1:]) + list(s_vals), color=colors[s], alpha=ALPHA, linewidth=LINEWIDTH)

            # 3. Plot Ground Truth (Target)
            # gt_vals = b_y[0, :][q_mask].cpu().numpy()[q_sort]
            # ax.scatter(q_times, gt_vals, color=COLOR_GT, marker="x", s=25, linewidth=1.5, label="Ground Truth", zorder=3)
            # ax.plot(o_times, o_vals, color=COLOR_OBS, marker="x", markersize=1, linewidth=1.5)

            # 4. Plot Observations (History)
            # ax.scatter(o_times, o_vals, color=COLOR_OBS, marker="o", s=30, edgecolor="white", linewidth=0.5, label="Observations", zorder=4)

            # Labels

            # Draw black box around plot area instead of x and y axes
            ax.spines["bottom"].set_visible(True)
            ax.spines["bottom"].set_color("black")
            ax.spines["left"].set_visible(True)
            ax.spines["left"].set_color("black")
            ax.spines["right"].set_visible(True)
            ax.spines["right"].set_color("black")
            ax.spines["top"].set_visible(True)
            ax.spines["top"].set_color("black")

            # ylab = channel_names[channel] if channel_names and len(channel_names) > channel else f"Ch. {channel}"

            # ax.set_ylabel(ylab)
            # x.grid(True, linestyle="--", alpha=0.4)
            ax.set_xticks([])
            ax.set_yticks([])

        # Legend & Save
        axes[-1].set_xlabel("Time $t$")
        h, l = axes[0].get_legend_handles_labels()
        # fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3, frameon=False)

        save_file = f"{path}_inference_inst{instance_idx}.pdf"
        fig.savefig(save_file, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved inference plot: {save_file}")


USHCN_CHANNELS = ["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]


def plot_samples(path, model, dataloader, n_samples=10, n_instances=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with torch.no_grad():

        # Get a single batch from test loader
        test_batch = next(iter(dataloader))
        obs, mobs, xq, mq, y = (tensor.to(model.device) for tensor in test_batch)
        jnll = model.jnll(obs=obs, mobs=mobs, xq=xq, mq=mq, y=y)
        norm_jnll = jnll / mq.sum(-1)
        print(norm_jnll)
        if xq.shape[1] > 0:
            # Generate samples from the model
            samples = model.sample(obs=obs, mobs=mobs, xq=xq, mq=mq, n_samples=n_samples)

            # Plot the results

            # Select the first sample from the batch for plotting
            for sample_idx in range(n_instances):
                n_channels = int(torch.max(obs[:, :, 1]).item()) + 1
                n_samples_to_plot = n_samples  # Plot up to 10 samples
                fig, axes = plt.subplots(n_channels, 1, figsize=(12, 3 * n_channels))
                # Display the joint negative log-likelihood for this instance on the figure
                try:
                    jnll_val = float(norm_jnll[sample_idx].detach().cpu().item())
                except Exception:
                    jnll_val = float(norm_jnll.detach().cpu().item())
                fig.suptitle(f"Joint NLL (instance {sample_idx}): {jnll_val:.3f}", fontsize=14)
                if n_channels == 1:
                    axes = [axes]

                for channel in range(n_channels):
                    ax = axes[channel]

                    # Plot observations (conditioning data)
                    obs_mask = torch.logical_and(mobs[sample_idx, :].bool(), (obs[sample_idx, :, 1] == channel))
                    obs_times = obs[sample_idx, :, 0][obs_mask].cpu().numpy()
                    obs_values = obs[sample_idx, :, -1][obs_mask].cpu().numpy()
                    ax.scatter(obs_times, obs_values, color="blue", label="Observations", s=30, alpha=0.7)

                    # Plot query points (ground truth)
                    query_mask = torch.logical_and(mq[sample_idx, :].bool(), (xq[sample_idx, :, 1] == channel))
                    query_times = xq[sample_idx, :, 0][query_mask].cpu().numpy()
                    query_values = y[sample_idx, :][query_mask].cpu().numpy()
                    ax.scatter(query_times, query_values, color="red", label="Ground Truth", s=30, alpha=0.7)

                    # Plot samples
                    for sample_num in range(n_samples_to_plot):
                        sample_values = samples[sample_idx, :, sample_num][query_mask].cpu().numpy()
                        colors = plt.cm.tab10(sample_num / n_samples_to_plot)  # Generate different colors
                        ax.plot(query_times, sample_values, color=colors, alpha=0.3, marker="x")
                        # if len(sample_values) > 0:
                        #    ax.set_ylim(sample_values.min() - 0.5, sample_values.max() + 0.5)

                    # Add sample legend only once
                    ax.set_xlabel("Time")
                    if n_channels == 5:
                        ax.set_ylabel(USHCN_CHANNELS[channel])
                    else:
                        ax.set_ylabel(f"Channel {channel}")
                ax.legend()
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(f"{path}_{sample_idx}.png", dpi=300, bbox_inches="tight")
                plt.clf()
