#!/usr/bin/env python3

import argparse
import random
import sys
import time
from pathlib import Path

proj_folder = Path(__file__).resolve().parents[3]

# Add the project root to sys.path if not already there
if str(proj_folder) not in sys.path:
    sys.path.append(str(proj_folder))


import numpy as np
import torch
import torch.nn as nn
import torchinfo
from profiti.model import ProFITi
from profiti.utils import profiti_collate_fn
from profiti.utils.evaluate import Evaluator
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils import load_data
from src.utils.plot import plot_instance_inference
from src.utils.utils import get_gpu_memory_allocated


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Training Script for ProFITi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training parameters
    training_group = parser.add_argument_group("Training Parameters")
    training_group.add_argument("--epochs", "-e", type=int, default=2000, help="Maximum number of epochs")
    training_group.add_argument("--batch-size", "-bs", type=int, default=64, help="Training batch size")
    parser.add_argument("-vbs", "--val_batch_size", default=64, type=int, help="Validation batch size")
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
    training_group.add_argument("--seed", "-s", type=int, default=42, help="Random seed for reproducibility")
    training_group.add_argument("--experiment_id", type=int, default=None, help="if given load the model")

    # Model parameters
    model_group = parser.add_argument_group("Model Parameters")
    model_group.add_argument("--hidden-size", "-hs", type=int, default=32, help="Hidden layer size")
    model_group.add_argument("--nlayers", "-nl", type=int, default=4, help="Number of layers")
    model_group.add_argument("--attn-head", "-ahd", type=int, default=1, help="Number of attention heads")
    model_group.add_argument("--latent-dim", "-ldim", type=int, default=64, help="Latent dimension size")
    model_group.add_argument("--flayers", "-fl", type=int, default=7, help="Number of flow layers")
    model_group.add_argument(
        "--marginal-training",
        "-mt",
        action="store_true",
        default=False,
        help="Use marginal training objective",
    )

    # Dataset parameters
    data_group = parser.add_argument_group("Dataset Parameters")
    data_group.add_argument("--dataset", "-dset", type=str, default="ushcn")
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

    # Additional parameters
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--output-dir",
        type=str,
        default="saved_models",
        help="Directory to save models",
    )
    misc_group.add_argument("--early-stop-patience", type=int, default=30, help="Early stopping patience")
    misc_group.add_argument("--scheduler-patience", type=int, default=10, help="Scheduler patience")
    misc_group.add_argument("--plot", action="store_true")

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
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Enable anomaly detection for debugging
    # torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


class Trainer:
    """Elegant trainer class for ProFITi model."""

    def __init__(
        self,
        model: nn.Module,
        args,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
    ):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            # Move batch to device
            batch = [tensor.to(self.device) for tensor in batch]
            tx, cx, x, mx, tq, cq, y, mq = batch
            if not mq.any():
                continue
            # Forward pass
            self.optimizer.zero_grad()
            self.model.distribution(tx, cx, x, mx, tq, cq, mq)
            if self.args.marginal_training:
                mnll = self.model.compute_mnll(y, mq)
                loss = mnll  # mnll is already averaged over all the queries
                total_loss += mnll.item()
                total_samples += mq.sum().item()
            else:
                njnll = self.model.compute_njnll(y, mq)
                loss = njnll.mean()  # njnll is not averaged hence we take mean
                total_loss += njnll.sum().item()
                total_samples += tx.shape[0]

            # Backward pass
            alloc = get_gpu_memory_allocated()
            loss.backward()
            self.optimizer.step()

            # Accumulate statistics

        return total_loss / total_samples, alloc

    def evaluate(self, data_loader) -> float:
        """Evaluate model on given data loader."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                batch = [tensor.to(self.device) for tensor in batch]
                tx, cx, x, mx, tq, cq, y, mq = batch
                if not mq.any():
                    continue
                # Forward pass
                self.optimizer.zero_grad()
                self.model.distribution(tx, cx, x, mx, tq, cq, mq)
                njnll = self.model.compute_njnll(y, mq)
                total_loss += njnll.sum().item()  # njnll is not averaged hence we take sum
                total_samples += tx.shape[0]
        return total_loss / total_samples

    def evaluate_mnll(self, data_loader):
        """Evaluate model on given data loader."""
        self.model.eval()
        total_samples = 0

        with torch.no_grad():
            total_mnll = 0
            for batch in data_loader:
                batch_ = [tensor.to(self.device) for tensor in batch]
                tx, cx, x, mx, tq_full, cq_full, y_full, mq_full = batch_
                if not mq_full.any():
                    continue
                # Forward pass
                self.optimizer.zero_grad()
                # pdb.set_trace()
                for i in range(tq_full.shape[1]):
                    tq = tq_full[:, i : i + 1]
                    cq = cq_full[:, i : i + 1]
                    y = y_full[:, i : i + 1]
                    mq = mq_full[:, i : i + 1]

                    self.model.distribution(tx, cx, x, mx, tq, cq, mq)
                    mnll = self.model.compute_mnll(y, mq)
                    total_mnll += mnll.item()
                total_samples += mq_full.sum().item()
        return total_mnll / total_samples

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
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        return checkpoint


def main():
    """Main training function."""
    # Setup
    args = parse_arguments()
    device = setup_environment(args.seed)

    print(f"Arguments: {args}")
    if args.experiment_id == None:
        experiment_id = int(time.time() * 1000) % 10000000
        print(f"Starting training with experiment ID: {experiment_id}")
    else:
        args.epochs = 0
        experiment_id = args.experiment_id
        print("Skip Training")

    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load dataset and create dataloaders
    task = load_data.load_dataset(args)
    train_loader, valid_loader, test_loader, test_loader_sampling = load_data.data_loaders(task, args, collate_fn=profiti_collate_fn)

    # Initialize model
    model_config = {
        "input_dim": task.dataset.shape[-1],
        "attn_head": args.attn_head,
        "latent_dim": args.latent_dim,
        "n_layers": args.nlayers,
        "f_layers": args.flayers,
        "marginal_training": args.marginal_training,
        "device": device,
    }

    model = ProFITi(**model_config).to(device)
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
        verbose=True,
    )

    # Initialize trainer
    trainer = Trainer(model, args, optimizer, scheduler, device)

    # Generate experiment ID and model path
    model_path = output_dir / f"{args.dataset}_fold{args.fold}_{experiment_id}.pt"
    plot_path = f"plots/{args.dataset}/profiti/{experiment_id}/"

    # Training loop
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        if (epoch - 1) % 20 == 0 and args.plot:
            with torch.no_grad():
                batch = next(iter(test_loader_sampling))
                plot_instance_inference(
                    model=model,
                    batch=batch,
                    path=f"{plot_path}/EPOCH={epoch}.png",
                    n_samples=50,
                    model_name="ProFITi",
                    color="#356208",
                )

        # Train
        train_loss, alloc = trainer.train_epoch(train_loader)
        epoch_time = time.time() - start_time

        # Validate
        val_loss = trainer.evaluate(valid_loader)

        print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {epoch_time:.2f}s")
        print(f"Alloc: {alloc:.3f} GB")
        if args.effi:
            sys.exit()

        # Save best model and check early stopping
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            trainer.save_checkpoint(str(model_path), epoch, args)
            trainer.early_stop_counter = 0
            print(f"New best model saved with val_loss: {val_loss:.6f}")
        else:
            trainer.early_stop_counter += 1

        # Early stopping
        if trainer.early_stop_counter >= args.early_stop_patience:
            print(f"Early stopping after {args.early_stop_patience} epochs without improvement")
            break

        # Update scheduler
        scheduler.step(val_loss)

    # Final evaluation on test set
    print("Loading best model for final evaluation...")
    trainer.load_checkpoint(str(model_path))

    start_time = time.time()
    test_njnll = trainer.evaluate(test_loader)
    eval_time = time.time() - start_time
    test_mnll = trainer.evaluate_mnll(test_loader)
    print("Final Results:")
    print(f"Best Val Loss: {trainer.best_val_loss:.6f}")
    print(f"Test Loss: {test_njnll:.6f}")
    print(f"Evaluation Time: {eval_time:.2f}s")
    print(f"best_val_loss: {trainer.best_val_loss:.6f}")
    print(f"test_time: {eval_time:.2f}")
    print(f"test_njnll: {test_njnll:.6f}")
    print(f"test_mnll: {test_mnll:.6f}")

    evaluater = Evaluator(
        model=model,
        args=args,
        device=device,
    )
    mse, _, crps, energy_score, _, _, _ = evaluater.evaluate_additional_metrics(data_loader=test_loader_sampling, nsamples=1000)
    print(f"test_mse: {mse:.6f}")
    print(f"test_crps: {crps:.6f}")
    print(f"test_energy_score: {energy_score:.6f}")

    return {
        "best_val_loss": trainer.best_val_loss,
        "test_loss": test_njnll,
        "experiment_id": experiment_id,
    }


if __name__ == "__main__":
    print(" ".join(sys.argv))
    main()
