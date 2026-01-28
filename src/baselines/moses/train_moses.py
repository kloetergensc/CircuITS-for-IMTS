import argparse
import random
import sys
from pathlib import Path

proj_folder = Path(__file__).resolve().parents[3]

# Add the project root to sys.path if not already there
if str(proj_folder) not in sys.path:
    sys.path.append(str(proj_folder))

import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchinfo
from core.model import Moses
from core.utils.metrics import AdditionalLossesComputer, compute_likelihood_losses
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils import load_data
from src.utils.plot import plot_instance_inference
from src.utils.utils import get_gpu_memory_allocated


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Training Script for Marginal Normalizing Flows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training parameters
    training_group = parser.add_argument_group("Training Parameters")
    training_group.add_argument("--epochs", "-e", type=int, default=1000, help="Maximum number of epochs")
    training_group.add_argument("--batch-size", "-bs", type=int, default=64, help="Training batch size")
    training_group.add_argument("--val-batch-size", "-vbs", type=int, default=10, help="Validation batch size")
    training_group.add_argument("--learn-rate", "-lr", type=float, default=1e-3, help="Learning rate")
    training_group.add_argument(
        "--betas",
        "-b",
        nargs=2,
        type=float,
        default=[0.9, 0.999],
        help="Adam optimizer betas",
    )
    training_group.add_argument(
        "--weight-decay",
        "-wd",
        type=float,
        default=1e-3,
        help="Weight decay for regularization",
    )
    training_group.add_argument("--seed", "-s", type=int, default=None, help="Random seed for reproducibility")

    # Model parameters
    model_group = parser.add_argument_group("Model Parameters")
    model_group.add_argument("--flayers", "-fl", type=int, default=1, help="Number of flow layers")
    model_group.add_argument(
        "--n-gaussians",
        "-ng",
        type=int,
        default=3,
        help="Number of Gaussian components",
    )
    model_group.add_argument("--n-heads", "-nh", type=int, default=4, help="Number of attention heads")
    model_group.add_argument(
        "--encoder",
        type=str,
        default="transformer",
        choices=["transformer", "grafiti"],
        help="Which encoder to use: transformer or grafiti",
    )
    model_group.add_argument(
        "--enc-layers",
        type=int,
        default=1,
        help="Number of encoder layers for self attention",
    )
    model_group.add_argument("--n-hiddens", type=int, default=128, help="Number of hidden dimensions")

    # Dataset parameters
    data_group = parser.add_argument_group("Dataset Parameters")
    data_group.add_argument(
        "--dataset",
        "-dset",
        type=str,
        default="ushcn",
        choices=["ushcn", "mimiciii", "mimiciv", "physionet2012", "bifurcation", "brownian"],
        help="Dataset to use",
    )
    data_group.add_argument("--forc-time", "-ft", type=int, default=0, help="Forecast horizon in hours")
    data_group.add_argument("--cond-time", "-ct", type=int, default=36, help="Conditioning range in hours")
    data_group.add_argument(
        "--nfolds",
        "-nf",
        type=int,
        default=5,
        help="Number of folds for cross-validation",
    )
    data_group.add_argument("--fold", "-f", type=int, default=0, help="Current fold number")

    # Spline parameters
    spline_group = parser.add_argument_group("splines")
    spline_group.add_argument("--bounds", type=int, default=20, help="bound for the spline")
    spline_group.add_argument("--num-bins", type=int, default=16, help="number of bins in the spline")

    # Additional parameters
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--output-dir",
        type=str,
        default="saved_models",
        help="Directory to save models",
    )
    misc_group.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    misc_group.add_argument("--scheduler-patience", type=int, default=10, help="Scheduler patience")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--plot", action="store_true")

    parser.add_argument("--channel", default=4, type=int, help="only use together with the brownian synthetic dataset")
    parser.add_argument("--timesteps", default=50, type=int, help="only use together with the brownian synthetic dataset")
    parser.add_argument("--effi", action="store_true")

    return parser.parse_args()


def setup_environment(seed: int) -> torch.device:
    """Setup environment and random seeds."""
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # CUDA optimizations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


class Trainer:
    """Trainer class for Marginal Normalizing Flows model."""

    def __init__(
        self,
        model: nn.Module,
        args,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
    ):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_njnll = 0.0
        total_mnll = 0.0
        total_instances = 0
        total_queries = 0
        for batch in train_loader:
            # Move batch to device and unpack
            obs, mobs, qry, mq, y = (tensor.to(self.device) for tensor in batch)
            tobs = obs[:, :, 0]
            cobs = obs[:, :, 1]
            xobs = obs[:, :, 2]
            tq = qry[:, :, 0]
            cq = qry[:, :, 1]

            # Forward pass
            self.optimizer.zero_grad()
            self.model(tobs, cobs, mobs, xobs, tq, cq, mq)

            # Compute loss
            n_instances = mq.shape[0]
            n_queries = mq.sum().item()
            njnll, mnll = compute_likelihood_losses(self.model, y, mq)
            alloc = get_gpu_memory_allocated()

            # Backward pass
            njnll.backward()
            self.optimizer.step()

            # Accumulate statistics
            total_njnll += njnll.item() * n_instances
            total_mnll += mnll.item() * n_queries
            total_instances += n_instances
            total_queries += n_queries

        return total_njnll / total_instances, total_mnll / total_queries, alloc

    def evaluate(self, eval_loader) -> Tuple[float, float]:
        """Evaluate model on given data loader."""
        self.model.eval()
        total_mnll = 0.0
        total_njnll = 0.0
        total_instances = 0
        total_queries = 0

        with torch.no_grad():
            for batch in eval_loader:
                # Move batch to device and unpack
                obs, mobs, qry, mq, y = (tensor.to(self.device) for tensor in batch)
                tobs = obs[:, :, 0]
                cobs = obs[:, :, 1]
                xobs = obs[:, :, 2]
                tq = qry[:, :, 0]
                cq = qry[:, :, 1]

                # Forward pass
                self.optimizer.zero_grad()
                self.model(tobs, cobs, mobs, xobs, tq, cq, mq)

                # Compute loss
                n_instances = mq.shape[0]
                n_queries = mq.sum().item()
                njnll, mnll = compute_likelihood_losses(self.model, y, mq)

                # Accumulate statistics
                total_njnll += njnll.item() * n_instances
                total_mnll += mnll.item() * n_queries
                total_instances += n_instances
                total_queries += n_queries

        return total_njnll / total_instances, total_mnll / total_queries

    def evaluate_for_all_metrics(self, eval_loader) -> Tuple[Dict, Dict, Dict]:
        """Evaluate model on given data loader."""
        self.model.eval()
        total_mnll = 0.0
        total_njnll = 0.0
        total_mse = 0.0
        total_robust_mse = 0.0
        total_crps = 0.0
        total_energy_score = 0.0
        total_instances = 0
        total_queries = 0
        compute_additional_metrics = AdditionalLossesComputer().compute_additional_metrics
        with torch.no_grad():
            for batch in eval_loader:
                # Move batch to device and unpack
                obs, mobs, qry, mq, y = (tensor.to(self.device) for tensor in batch)
                tobs = obs[:, :, 0]
                cobs = obs[:, :, 1]
                xobs = obs[:, :, 2]
                tq = qry[:, :, 0]
                cq = qry[:, :, 1]

                # Forward pass
                self.optimizer.zero_grad()
                self.model(tobs, cobs, mobs, xobs, tq, cq, mq)

                # Compute loss
                n_instances = mq.shape[0]
                n_queries = mq.sum().item()
                njnll, mnll = compute_likelihood_losses(self.model, y, mq)
                additional_metrics = compute_additional_metrics(model=self.model, y=y, mq=mq, n_samples=100)

                # Accumulate statistics
                total_njnll += njnll.item() * n_instances
                total_mnll += mnll.item() * n_queries
                total_mse += additional_metrics.mse.item() * n_queries
                total_robust_mse += additional_metrics.robust_mse.item() * n_queries
                total_crps += additional_metrics.crps.item() * n_queries
                total_energy_score += additional_metrics.energy_score.item() * n_instances
                total_instances += n_instances
                total_queries += n_queries

            likelihood_metrics = {
                "njnll": total_njnll / total_instances,
                "mnll": total_mnll / total_queries,
            }
            energy_metrics = {
                "energy_score": total_energy_score / total_instances,
                "crps": total_crps / total_queries,
            }
            point_metrics = {
                "mse": total_mse / total_queries,
                "robust_mse": total_robust_mse / total_queries,
            }

        return likelihood_metrics, energy_metrics, point_metrics

    def save_checkpoint(self, filepath: str, epoch: int, args: argparse.Namespace):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        """Save model checkpoint."""
        torch.save(
            {
                "args": args,
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])


def main():
    """Main training function."""
    # Setup
    args = parse_arguments()
    experiment_id = int(time.time() * 1000) % 10000000
    if args.seed is None:
        args.seed = experiment_id
    device = setup_environment(args.seed)

    print(f"Arguments: {args}")

    print(f"Starting training with experiment ID: {experiment_id}")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load dataset and create dataloaders
    task = load_data.load_dataset(args)
    train_loader, valid_loader, test_loader, _ = load_data.data_loaders(task, args)

    # Initialize model
    model_config = {
        "n_inputs": task.dataset.shape[-1],
        "num_components": args.n_gaussians,
        "latent_dim": args.n_hiddens,
        "num_flow_layers": args.flayers,
        "num_encoder_layers": args.enc_layers,
        "n_heads": args.n_heads,
        "bounds": args.bounds,
        "num_bins": args.num_bins,
        "encoder_model": args.encoder,
        "device": device,
    }

    model = Moses(**model_config).to(device)

    model.zero_grad(set_to_none=True)

    # Print model summary
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    print(f"Model summary:\n{torchinfo.summary(model)}")

    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learn_rate,
        betas=args.betas,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.scheduler_patience,
        factor=0.5,
        min_lr=1e-5,
    )

    # Initialize trainer
    trainer = Trainer(model, args, optimizer, scheduler, device)

    # Generate model path
    model_path = output_dir / f"mnf_{args.dataset}_fold{args.fold}_{experiment_id}.pt"

    # Training loop
    print("Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_njnll, train_mnll, alloc = trainer.train_epoch(train_loader)
        # Validate
        epoch_time = time.time() - start_time
        val_njnll, val_mnll = trainer.evaluate(valid_loader)

        print(f"Epoch {epoch:3d}/{args.epochs} | Train njNLL: {train_njnll:.6f} | Train mNLL: {train_mnll:.6f} | " f"Val njNLL: {val_njnll:.6f} | Val mNLL: {val_mnll:.6f} | Time: {epoch_time:.2f}s")
        print(f"Alloc: {alloc:.3f} GB")
        if args.effi:
            sys.exit()
        # Save best model and check early stopping
        if val_njnll < trainer.best_val_loss:
            trainer.best_val_loss = val_njnll
            trainer.save_checkpoint(str(model_path), epoch, args)
            trainer.early_stop_counter = 0
            print(f"New best model saved with val_loss: {val_njnll:.6f}")
        else:
            trainer.early_stop_counter += 1

        # Early stopping
        if trainer.early_stop_counter >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement")
            break

        # Update scheduler
        scheduler.step(val_njnll)

    # Final evaluation on test set
    print("Loading best model for final evaluation...")
    trainer.load_checkpoint(str(model_path))
    with torch.no_grad():
        if args.plot:
            batch = next(iter(test_loader))
            plot_instance_inference(
                model=model,
                path=f"plots/{args.dataset}/MOSES-{args.n_gaussians}/{experiment_id}/",
                color="#DD7514",
                model_name=f"MOSES-{args.n_gaussians}",
                n_samples=50,
                batch=batch,
            )
            model.to(device)
        trainer.load_checkpoint(str(model_path))

        start_time = time.time()
        likelihood_metrics, energy_metrics, point_metrics = trainer.evaluate_for_all_metrics(test_loader)
        eval_time = time.time() - start_time

    print("Final Results:")
    print(f"best_val_loss: {trainer.best_val_loss:.6f}")
    print(f"test_njnll: {likelihood_metrics['njnll']:.6f}")
    print(f"test_mnll: {likelihood_metrics['mnll']:.6f}")
    print(f"test_mse: {point_metrics['mse']:.6f}")
    print(f"Test robust mse: {point_metrics['robust_mse']:.6f}")
    print(f"test_crps: {energy_metrics['crps']:.6f}")
    print(f"test_energy_score: {energy_metrics['energy_score']:.6f}")
    print(f"Evaluation Time: {eval_time:.2f}s")


if __name__ == "__main__":
    print(" ".join(sys.argv))
    main()
    exit()
