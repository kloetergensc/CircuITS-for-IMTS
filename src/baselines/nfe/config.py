"""
Configuration Module

Contains all configuration constants, argument parsing, and environment setup.
"""

import argparse
import logging
import os
import random
import warnings

import numpy as np
import torch

# =============================================================================
# CONSTANTS
# =============================================================================

EARLY_STOPPING_PATIENCE = 30
MAX_TRAINING_TIME_SECONDS = 86400  # 24 hours
INITIAL_BEST_LOSS = 1e8
EXPERIMENT_ID_RANGE = 10000000

ODE_DATASETS = [
    "FitzHugh1961_NerveMembrane",
    "nutrient_digestion",
    "skeletal_muscle",
    "volterra",
    "Midmyocardial",
    "hypoglossal_motoneuron",
    "insulin_glucose",
]


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================


def setup_environment():
    """Configure environment variables and backend settings."""
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Logging setup
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings(action="ignore", category=UserWarning, module="torch")


def setup_random_seeds(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value (None to skip seeding)
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        print(f"Random seed set to: {seed}")


# =============================================================================
# ARGUMENT PARSER
# =============================================================================


def create_argument_parser():
    """
    Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser("Neural flows")

    # Basic experiment configuration
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--experiment",
        type=str,
        help="Which experiment to run",
        choices=["latent_ode", "synthetic", "gru_ode_bayes", "tpp", "stpp"],
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model type: ODE, flow-based, or RNN",
        default="flow",
        choices=["ode", "flow", "rnn"],
    )
    parser.add_argument("--dataset", type=str, default="ushcn", help="Dataset name")

    # Training hyperparameters
    _add_training_arguments(parser)

    # Neural network architecture
    _add_network_arguments(parser)

    # ODE-specific parameters
    _add_ode_arguments(parser)

    # Flow model parameters
    _add_flow_arguments(parser)

    # Task-specific parameters
    _add_task_specific_arguments(parser)

    return parser


def _add_training_arguments(parser):
    """Add training-related arguments."""
    parser.add_argument("--epochs", type=int, default=1, help="Maximum training epochs")
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Early stopping patience",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0, help="Weight decay (L2 regularization)")
    parser.add_argument(
        "--lr-scheduler-step",
        type=int,
        default=-1,
        help="Steps between learning rate decay",
    )
    parser.add_argument("--lr-decay", type=float, default=0.9, help="Learning rate decay factor")
    parser.add_argument(
        "-b",
        "--betas",
        default=(0.9, 0.999),
        type=float,
        nargs=2,
        help="Adam optimizer beta parameters",
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=50, help="Training batch size")
    parser.add_argument("--clip", type=float, default=1, help="Gradient clipping threshold")
    parser.add_argument("-f", "--fold", default=2, type=int, help="Cross-validation fold number")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Evaluation batch size")


def _add_network_arguments(parser):
    """Add neural network architecture arguments."""
    parser.add_argument("--hidden-layers", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("--hidden-dim", type=int, default=1, help="Hidden layer dimension")
    parser.add_argument(
        "--activation",
        type=str,
        default="Tanh",
        help="Hidden layer activation function",
    )
    parser.add_argument(
        "--final-activation",
        type=str,
        default="Identity",
        help="Final layer activation function",
    )


def _add_ode_arguments(parser):
    """Add ODE-specific arguments."""
    parser.add_argument(
        "--odenet",
        type=str,
        default="concat",
        choices=["concat", "gru"],
        help="Type of ODE network",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="dopri5",
        choices=["dopri5", "rk4", "euler"],
        help="ODE solver algorithm",
    )
    parser.add_argument("--solver-step", type=float, default=0.05, help="Fixed solver step size")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for ODE solver")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for ODE solver")


def _add_flow_arguments(parser):
    """Add flow model arguments."""
    parser.add_argument(
        "--flow-model",
        type=str,
        default="gru",
        choices=["coupling", "resnet", "gru"],
        help="Flow model architecture",
    )
    parser.add_argument("--flow-layers", type=int, default=1, help="Number of flow layers")
    parser.add_argument(
        "--time-net",
        type=str,
        default="TimeLinear",
        choices=["TimeFourier", "TimeFourierBounded", "TimeLinear", "TimeTanh"],
        help="Time embedding network",
    )
    parser.add_argument(
        "--time-hidden-dim",
        type=int,
        default=1,
        help="Time feature dimension (for Fourier)",
    )


def _add_task_specific_arguments(parser):
    """Add task-specific arguments for different experiments."""
    # Latent ODE specific
    parser.add_argument(
        "--classify",
        type=int,
        default=0,
        choices=[0, 1],
        help="Include classification loss",
    )
    parser.add_argument(
        "--extrap",
        type=int,
        default=0,
        choices=[0, 1],
        help="Extrapolation mode (vs interpolation)",
    )
    parser.add_argument("-n", type=int, default=10000, help="Dataset size (latent_ode)")
    parser.add_argument(
        "--quantization",
        type=float,
        default=0.016,
        help="Quantization for physionet dataset",
    )
    parser.add_argument("--latents", type=int, default=20, help="Latent state dimension")
    parser.add_argument("--rec-dims", type=int, default=20, help="Recognition model dimensionality")
    parser.add_argument("--gru-units", type=int, default=100, help="GRU units per layer")
    parser.add_argument("--timepoints", type=int, default=100, help="Total number of time points")
    parser.add_argument("--max-t", type=float, default=5.0, help="Maximum time for subsampling")

    # GRU-ODE-Bayes specific
    parser.add_argument("--mixing", type=float, default=0.0001, help="KL vs update loss ratio")
    parser.add_argument(
        "--gob_prep_hidden",
        type=int,
        default=10,
        help="Hidden state size for preprocessing",
    )
    parser.add_argument(
        "--gob_cov_hidden",
        type=int,
        default=50,
        help="Hidden state size for covariates",
    )
    parser.add_argument(
        "--gob_p_hidden",
        type=int,
        default=25,
        help="Hidden state size for initialization",
    )
    parser.add_argument(
        "--invertible",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether network is invertible",
    )

    # TPP (Temporal Point Process) specific
    parser.add_argument("--components", type=int, default=8, help="Number of mixture components")
    parser.add_argument(
        "--decoder",
        type=str,
        default="continuous",
        choices=["continuous", "mixture"],
        help="Intensity function type",
    )
    parser.add_argument("--rnn", type=str, choices=["gru", "lstm"], help="RNN encoder type")
    parser.add_argument("--marks", type=int, default=0, choices=[0, 1], help="Use marked TPP")

    # STPP (Spatio-Temporal Point Process) specific
    parser.add_argument(
        "--density-model",
        type=str,
        choices=["independent", "attention", "jump"],
        help="Density model type",
    )
    parser.add_argument("-ft", "--forc-time", default=6, type=int, help="Forecast horizon (hours)")
    parser.add_argument("-ct", "--cond-time", default=36, type=int, help="Conditioning range (hours)")
    parser.add_argument("-nf", "--nfolds", default=5, type=int, help="Number of cross-validation folds")


# Initialize environment on module import
setup_environment()
