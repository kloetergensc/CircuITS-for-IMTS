"""
Training and Evaluation Module

Contains all training loop logic, evaluation metrics, and checkpoint management.
"""

import pdb
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import properscoring as ps
import torch
from nfe.config import INITIAL_BEST_LOSS, MAX_TRAINING_TIME_SECONDS
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast

# =============================================================================
# LOSS FUNCTIONS AND METRICS
# =============================================================================

# Gaussian negative log-likelihood loss
gaussian_nll_loss = torch.nn.GaussianNLLLoss(full=True, reduction="none")


def compute_mse(targets: Tensor, predictions: Tensor, mask: Tensor) -> Tensor:
    """
    Compute Mean Squared Error.

    Args:
        targets: Ground truth values
        predictions: Model predictions (mean, logvar)
        mask: Binary mask for valid observations

    Returns:
        MSE loss value
    """
    pred_mean = predictions[0]
    error = torch.sum((targets[mask] - pred_mean[mask]) ** 2)
    return error


def compute_nll(targets: Tensor, predictions: Tensor, mask: Tensor, lengths: list) -> Tensor:
    """
    Compute batch-averaged Negative Log-Likelihood.

    Args:
        targets: Ground truth values
        predictions: Tuple of (mean, logvar) predictions
        mask: Binary mask for valid observations
        lengths: List of sequence lengths per sample

    Returns:
        Averaged NLL across batch
    """
    pred_mean, pred_logvar = predictions
    errors = gaussian_nll_loss(pred_mean, targets, pred_logvar.exp()) * mask

    nll_sum = 0
    batch_count = 0
    cumulative_lengths = np.cumsum(lengths)
    start_idx = 0
    # pdb.set_trace()
    for i, length in enumerate(lengths):
        if length == 0:
            continue

        batch_count += 1
        end_idx = cumulative_lengths[i]

        sample_errors = errors[start_idx:end_idx]
        sample_mask = mask[start_idx:end_idx]
        mask_sum = sample_mask.sum() + 1e-8

        nll_sum += sample_errors.sum() / mask_sum
        start_idx = end_idx

    return nll_sum


def compute_energy_score(targets: torch.Tensor, predictions: tuple, mask: torch.Tensor, lengths: list, num_samples: int = 1000) -> torch.Tensor:
    """
    Compute batch-averaged Multivariate Energy Score using Monte Carlo sampling.

    This treats the entire sequence (or flattened features) as the multivariate variable.

    Args:
        targets: Ground truth values [Batch, SeqLen, Dim]
        predictions: Tuple of (mean, logvar) [Batch, SeqLen, Dim]
        mask: Binary mask for valid observations [Batch, SeqLen] or [Batch, SeqLen, Dim]
        lengths: List of sequence lengths
        num_samples: Number of MC samples to estimate expectations (k)

    Returns:
        Averaged Energy Score across batch
    """
    pred_mean, pred_logvar = predictions
    std = (0.5 * pred_logvar).exp()

    batch_scores = []

    # We iterate manually or use sophisticated masking to handle variable lengths correctly
    # for the Euclidean norm calculation.

    cumulative_lengths = np.cumsum(lengths)
    start_idx = 0

    for i, length in enumerate(lengths):
        if length == 0:
            continue

        end_idx = cumulative_lengths[i]

        # 1. Extract valid data for this sample
        # Slices are [Length, Dim]
        # We flatten to 1D vector [Length * Dim] to treat the whole trajectory as one multivariate point
        # (Or keep as [Length, Dim] if you want to compute norm over specific dims,
        # but standard Energy Score treats the whole "datum" as a vector).

        # Taking the relevant slice based on your snippet's structure
        # Assuming inputs are already flattened batch-wise or we slice using the 1D logic from your snippet:
        # NOTE: Your NLL snippet implies `targets` might be flattened [Total_Time_Steps, Dim].
        # I will adhere to that flattened logic to match your input format.

        curr_target = targets[start_idx:end_idx]  # [L, D]
        curr_mean = pred_mean[start_idx:end_idx]  # [L, D]
        curr_std = std[start_idx:end_idx]  # [L, D]

        # Flatten time and features into a single multivariate vector y
        y = curr_target.reshape(-1)  # [L*D]
        mu = curr_mean.reshape(-1)  # [L*D]
        sigma = curr_std.reshape(-1)  # [L*D]

        # 2. Generate MC samples: x ~ N(mu, sigma)
        # Shape: [num_samples, L*D]
        noise = torch.randn(num_samples, y.shape[0], device=targets.device)
        x_samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise

        # 3. Term 1: E[ ||x - y|| ]
        # dists_to_target: [num_samples]
        dists_to_target = torch.norm(x_samples - y.unsqueeze(0), p=2, dim=1)
        term1 = dists_to_target.mean()

        # 4. Term 2: -0.5 * E[ ||x - x'|| ]
        # We can estimate this efficiently by shuffling the samples or pairwise computation.
        # A simple estimator is to split samples in two halves and compute distance.
        # Alternatively, pairwise distance matrix (expensive for high num_samples).
        # Efficient approach: Permutation

        x_samples_1 = x_samples[: num_samples // 2]
        x_samples_2 = x_samples[num_samples // 2 : 2 * (num_samples // 2)]

        dists_inter_sample = torch.norm(x_samples_1 - x_samples_2, p=2, dim=1)
        term2 = 0.5 * dists_inter_sample.mean()

        # Energy Score
        sample_score = term1 - term2
        batch_scores.append(sample_score)

        start_idx = end_idx

    if len(batch_scores) == 0:
        return torch.tensor(0.0, device=targets.device)

    return torch.stack(batch_scores).sum()


def compute_mnll(targets: Tensor, predictions: Tensor, mask: Tensor, lengths: list) -> Tensor:
    """
    Compute total (non-averaged) Negative Log-Likelihood.

    Args:
        targets: Ground truth values
        predictions: Tuple of (mean, logvar) predictions
        mask: Binary mask for valid observations
        lengths: List of sequence lengths per sample

    Returns:
        Total NLL
    """
    pred_mean, pred_logvar = predictions
    errors = gaussian_nll_loss(pred_mean, targets, pred_logvar.exp()) * mask
    return errors.sum()


def compute_crps(targets: Tensor, predictions: Tensor, mask: Tensor) -> float:
    """
    Compute Continuous Ranked Probability Score.

    Args:
        targets: Ground truth values
        predictions: Tuple of (mean, logvar) predictions
        mask: Binary mask for valid observations

    Returns:
        CRPS value
    """
    pred_mean, pred_logvar = predictions
    pred_std = torch.exp(pred_logvar) ** 0.5

    targets_np = targets[mask].detach().cpu().numpy()
    mean_np = pred_mean[mask].detach().cpu().numpy()
    std_np = pred_std[mask].detach().cpu().numpy()

    crps_values = ps.crps_gaussian(targets_np, mean_np, std_np)
    return np.sum(crps_values)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def reorder_validation_data(targets, masks, time_steps):
    """
    Reorder validation data based on time step indices.

    This function reorganizes validation targets and masks so that all values
    at the same timestep across different samples are grouped together.

    Args:
        targets: Validation target values
        masks: Validation masks
        time_steps: Validation time steps per sample

    Returns:
        Tuple of (reordered_targets, reordered_masks) as lists
    """
    num_obs = torch.Tensor([len(times) for times in time_steps])
    max_timesteps = int(torch.max(num_obs).item())

    reordered_targets = []
    reordered_masks = []
    cumsum = torch.cat([torch.Tensor([0]), torch.cumsum(num_obs, dim=0)[:-1]])

    for timestep_idx in range(max_timesteps):
        # Find samples with data at this timestep
        valid_samples = num_obs > timestep_idx
        indices = (cumsum[valid_samples] + timestep_idx).long()

        reordered_targets.append(targets[indices])
        reordered_masks.append(masks[indices])

    return reordered_targets, reordered_masks


# =============================================================================
# MODEL EVALUATION
# =============================================================================


def evaluate_model(model, dataloader, args, device):
    """
    Evaluate model on a dataset.

    Args:
        model: Neural network model
        dataloader: DataLoader for evaluation
        args: Configuration arguments with solver_step
        device: Torch device (CPU/GPU)

    Returns:
        Tuple of (avg_nll, total_mnll, mse, crps) metrics:
        - avg_nll: Average negative log-likelihood per sample
        - total_mnll: Mean negative log-likelihood per observation
        - mse: Mean squared error per observation
        - crps: Continuous ranked probability score per observation
    """
    model.eval()

    total_nll = 0
    total_mnll = 0
    total_mse = 0
    total_crps = 0
    total_observations = 0
    total_samples = 0
    total_energy_score = 0

    with torch.no_grad():
        for batch_data in dataloader:
            # Skip batches without validation data
            if batch_data["X_val"] is None:
                continue

            # Forward pass through model
            _, _, _, _, _, prediction_path = model(
                batch_data["times"],
                batch_data["num_obs"],
                batch_data["X"].to(device),
                batch_data["M"].to(device),
                delta_t=args.solver_step,
                cov=batch_data["cov"].to(device),
                return_path=True,
                val_times=batch_data["times_val"],
            )

            # Extract mean and variance predictions
            pred_mean, pred_variance = torch.chunk(prediction_path, 2, dim=1)

            # Reorder targets and masks based on validation time steps
            targets, masks = reorder_validation_data(batch_data["X_val"], batch_data["M_val"], batch_data["times_val"])

            targets = torch.cat(targets).to(device)
            masks = torch.cat(masks).to(device)
            predictions = (pred_mean, pred_variance)
            lengths = [len(times) for times in batch_data["times_val"]]

            # Compute metrics
            total_nll += compute_nll(targets, predictions, masks, lengths).item()
            total_mnll += compute_mnll(targets, predictions, masks, lengths).item()
            total_mse += compute_mse(targets, predictions, masks).item()
            total_crps += compute_crps(targets, predictions, masks)
            total_energy_score += compute_energy_score(targets=targets, predictions=predictions, mask=masks, lengths=lengths).item()

            total_observations += masks.sum().item()
            total_samples += len(np.nonzero(lengths)[0])

    # Return averaged metrics
    avg_nll = total_nll / max(total_samples, 1)
    avg_mnll = total_mnll / max(total_observations, 1)
    avg_mse = total_mse / max(total_observations, 1)
    avg_crps = total_crps / max(total_observations, 1)
    total_energy_score = total_energy_score / max(total_samples, 1)

    return avg_nll, avg_mnll, avg_mse, avg_crps, total_energy_score


def evaluate_model_fast(model, dataloader, args, device):
    """
    Evaluate model on a dataset.

    Args:
        model: Neural network model
        dataloader: DataLoader for evaluation
        args: Configuration arguments with solver_step
        device: Torch device (CPU/GPU)

    Returns:
        Tuple of (avg_nll, total_mnll, mse, crps) metrics:
        - avg_nll: Average negative log-likelihood per sample
        - total_mnll: Mean negative log-likelihood per observation
        - mse: Mean squared error per observation
        - crps: Continuous ranked probability score per observation
    """
    model.eval()

    total_nll = 0
    total_mnll = 0
    total_observations = 0
    total_samples = 0

    with torch.no_grad():
        for batch_data in dataloader:
            # Skip batches without validation data
            if batch_data["X_val"] is None:
                continue

            # Forward pass through model
            _, _, _, _, _, prediction_path = model(
                batch_data["times"],
                batch_data["num_obs"],
                batch_data["X"].to(device),
                batch_data["M"].to(device),
                delta_t=args.solver_step,
                cov=batch_data["cov"].to(device),
                return_path=True,
                val_times=batch_data["times_val"],
            )

            # Extract mean and variance predictions
            pred_mean, pred_variance = torch.chunk(prediction_path, 2, dim=1)

            # Reorder targets and masks based on validation time steps
            targets, masks = reorder_validation_data(batch_data["X_val"], batch_data["M_val"], batch_data["times_val"])

            targets = torch.cat(targets).to(device)
            masks = torch.cat(masks).to(device)
            predictions = (pred_mean, pred_variance)
            lengths = [len(times) for times in batch_data["times_val"]]

            # Compute metrics
            total_nll += compute_nll(targets, predictions, masks, lengths).item()
            total_mnll += compute_mnll(targets, predictions, masks, lengths).item()
            total_observations += masks.sum().item()
            total_samples += len(np.nonzero(lengths)[0])

    # Return averaged metrics
    avg_nll = total_nll / max(total_samples, 1)
    avg_mnll = total_mnll / max(total_observations, 1)

    return (
        avg_nll,
        avg_mnll,
    )


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================


def train_epoch_mp(model, dataloader, optimizer, args, device):
    model.train()
    epoch_losses = []
    scaler = GradScaler()  # For mixed precision

    for batch_data in dataloader:
        optimizer.zero_grad()

        # Use automatic mixed precision
        with autocast():
            _, loss, _, _, _ = model(
                batch_data["times"],
                batch_data["num_obs"],
                batch_data["X"].to(device),
                batch_data["M"].to(device),
                delta_t=args.solver_step,
                cov=batch_data["cov"].to(device),
                val_times=batch_data["times_val"],
            )
            normalized_loss = loss / batch_data["M"].size(0)

        # Scaled backward pass
        scaler.scale(normalized_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_losses.append(normalized_loss.item())

    return np.mean(epoch_losses)


def train_epoch(model, dataloader, optimizer, args, device):
    """
    Train model for one epoch.

    Args:
        model: Neural network model
        dataloader: Training data loader
        optimizer: Optimizer instance
        args: Configuration arguments with solver_step
        device: Torch device

    Returns:
        Average training loss for the epoch

    Raises:
        RuntimeError: If model produces non-finite loss (collapsed)
    """
    model.train()
    epoch_losses = []

    for batch_data in dataloader:
        optimizer.zero_grad()

        # Forward pass
        _, loss, _, _, _ = model(
            batch_data["times"],
            batch_data["num_obs"],
            batch_data["X"].to(device),
            batch_data["M"].to(device),
            delta_t=args.solver_step,
            cov=batch_data["cov"].to(device),
            val_times=batch_data["times_val"],
        )

        # Normalize loss by batch size
        normalized_loss = loss / batch_data["M"].size(0)

        # Check for numerical stability
        if not torch.isfinite(normalized_loss).item():
            raise RuntimeError("Model collapsed! Loss is not finite.")

        # Backward pass and optimization
        normalized_loss.backward()
        optimizer.step()
        epoch_losses.append(normalized_loss.item())

    return np.mean(epoch_losses)


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================


def save_checkpoint(model, optimizer, args, epoch, loss, experiment_id):
    """
    Save model checkpoint to disk.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        args: Configuration arguments
        epoch: Current epoch number
        loss: Current loss value
        experiment_id: Unique experiment identifier
    """
    checkpoint_path = f"saved_models/{args.dataset}_{experiment_id}.h5"
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "ARGS": args,
            "epoch": epoch,
            "state_dict": deepcopy(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )


def load_checkpoint(model, args, experiment_id):
    """
    Load model checkpoint from disk.

    Args:
        model: Model to load weights into
        args: Configuration arguments
        experiment_id: Unique experiment identifier
    """
    checkpoint_path = f"saved_models/{args.dataset}_{experiment_id}.h5"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================


def train_model(model, dataloaders, optimizer, scheduler, args, device, experiment_id):
    """
    Main training loop with early stopping and validation.

    This function trains the model for multiple epochs, validating after each
    epoch and saving checkpoints when validation performance improves. Training
    stops when:
    - Early stopping patience is exceeded
    - Maximum epochs are reached
    - Maximum training time is exceeded

    Args:
        model: Neural network model to train
        dataloaders: Dictionary of train/valid/test DataLoaders
        optimizer: Optimizer instance
        args: Configuration arguments with epochs and patience
        device: Torch device (CPU/GPU)
        experiment_id: Unique experiment identifier for checkpointing
    """
    best_val_loss = INITIAL_BEST_LOSS
    early_stop_counter = 0
    training_start_time = time.time()

    print("=" * 80)
    print("Starting Training")
    print("=" * 80)
    print(model)
    print("=" * 80)

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            args=args,
            device=device,
        )

        epoch_duration = time.time() - epoch_start_time

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Duration: {int(epoch_duration)}s")

        # Validation
        val_nll, val_mnll = evaluate_model_fast(model, dataloaders["valid"], args, device)
        scheduler.step(val_nll)

        print(f"  Val NLL: {val_nll:.4f}, Val MNLL: {val_mnll:.4f}")

        # Save best model
        if val_nll < best_val_loss:
            best_val_loss = val_nll
            save_checkpoint(model, optimizer, args, epoch, train_loss, experiment_id)
            early_stop_counter = 0
            print("  ✓ New best model saved!")
        else:
            early_stop_counter += 1

        # Check stopping conditions
        total_training_time = time.time() - training_start_time
        should_stop, stop_reason = _check_stopping_conditions(early_stop_counter, epoch, total_training_time, args)

        if should_stop:
            _finalize_training(
                model,
                dataloaders["test"],
                args,
                device,
                experiment_id,
                stop_reason,
                best_val_loss,
            )
            break


def _check_stopping_conditions(early_stop_counter, epoch, training_time, args):
    """
    Check if training should stop.

    Returns:
        Tuple of (should_stop, reason)
    """
    if early_stop_counter >= args.patience:
        return True, f"Early stopping (no improvement for {args.patience} epochs)"
    elif epoch == args.epochs - 1:
        return True, "All epochs completed"
    elif training_time > MAX_TRAINING_TIME_SECONDS:
        return True, "Maximum training time reached (24 hours)"

    return False, ""


def _finalize_training(model, test_loader, args, device, experiment_id, stop_reason, best_val_loss):
    """
    Finalize training by loading best model and evaluating on test set.
    """
    print("\n" + "=" * 80)
    print(f"Training stopped: {stop_reason}")
    print("=" * 80)

    # Load best model and evaluate on test set
    load_checkpoint(model, args, experiment_id)

    test_start_time = time.time()
    test_nll, test_mnll, test_mse, test_crps, test_energy_score = evaluate_model(model, test_loader, args, device)
    test_duration = time.time() - test_start_time

    print(f"\nFinal Test Results:")
    print(f"test_njnll: {test_nll:.4f}")
    print(f"test_mnll: {test_mnll:.4f}")
    print(f"test_mse: {test_mse:.4f}")
    print(f"test_crps: {test_crps:.6f}")
    print(f"test_energy_score: {test_energy_score:.4f}")
    print(f"best_val_loss: {best_val_loss:.4f}")
    print(f"  Evaluation Time: {test_duration:.2f}s")
    print("=" * 80)
