"""
Main Training Script

This is the entry point for training neural flow models.
Usage: python train.py --dataset <dataset_name> --model <model_type> [options]
"""

import sys
from pathlib import Path
from random import SystemRandom

nfe_folder = Path(__file__).resolve().parents[1]
if str(nfe_folder) not in sys.path:
    sys.path.append(str(nfe_folder))
proj_folder = Path(__file__).resolve().parents[3]
if str(proj_folder) not in sys.path:
    sys.path.append(str(proj_folder))


import torch
from config import EXPERIMENT_ID_RANGE, create_argument_parser, setup_random_seeds
from data_utils_wrapper import create_data_loaders, load_dataset
from experiments.gru_ode_bayes.experiment import GOB
from torch.optim.lr_scheduler import ReduceLROnPlateau
from training import train_model


def main():
    """Main execution function."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Generate unique experiment ID
    experiment_id = int(SystemRandom().random() * EXPERIMENT_ID_RANGE)

    # Display configuration
    print("=" * 80)
    print("Neural Flow Training")
    print("=" * 80)
    print(f"Command: {' '.join(sys.argv)}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Arguments: {args}")
    print("=" * 80)

    # Setup
    setup_random_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.data = args.dataset

    print(f"Device: {device}")

    # Load data
    print("\nLoading dataset...")
    task = load_dataset(args)
    dataloaders = create_data_loaders(task, args)
    print(f"Dataset loaded: {args.dataset}")

    # Initialize model
    print("\nInitializing model...")
    sample_batch = next(iter(dataloaders["train"]))
    input_size = sample_batch["M"].shape[-1]

    model = GOB().get_model(args, input_size)
    model.to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=args.lr_decay, patience=5)

    # Train model
    train_model(
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        device=device,
        experiment_id=experiment_id,
    )


if __name__ == "__main__":
    main()
