"""
Data Loading Module

Handles dataset loading and dataloader creation for different dataset types.
"""

import os

from nfe import data_utils
from nfe.config import ODE_DATASETS


def load_dataset(args):
    """
    Load the appropriate dataset based on arguments.

    Args:
        args: Parsed command-line arguments containing:
            - dataset: Name of the dataset to load
            - cond_time: Conditioning time range (hours)
            - forc_time: Forecast horizon (hours)
            - nfolds: Number of cross-validation folds

    Returns:
        Dataset task object with train/val/test splits

    Raises:
        ValueError: If dataset name is not recognized
    """
    dataset_name = args.dataset

    if dataset_name == "ushcn":
        return _load_ushcn(args)
    elif dataset_name == "mimiciii":
        return _load_mimic_iii(args)
    elif dataset_name == "mimiciv":
        return _load_mimic_iv(args)
    elif dataset_name == "physionet2012":
        return _load_physionet(args)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_ushcn(args):
    """Load USHCN climate dataset."""
    from tsdm.tasks import USHCN_DeBrouwer2019

    return USHCN_DeBrouwer2019(
        normalize_time=True,
        condition_time=args.cond_time,
        forecast_horizon=args.forc_time,
        num_folds=args.nfolds,
    )


def _load_mimic_iii(args):
    """Load MIMIC-III medical dataset."""
    from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019

    return MIMIC_III_DeBrouwer2019(
        normalize_time=True,
        condition_time=args.cond_time,
        forecast_horizon=args.forc_time,
        num_folds=args.nfolds,
    )


def _load_mimic_iv(args):
    """Load MIMIC-IV medical dataset."""
    from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021

    return MIMIC_IV_Bilos2021(
        normalize_time=True,
        condition_time=args.cond_time,
        forecast_horizon=args.forc_time,
        num_folds=args.nfolds,
    )


def _load_physionet(args):
    """Load Physionet 2012 medical dataset."""
    from tsdm.tasks.physionet2012 import Physionet2012

    return Physionet2012(
        normalize_time=True,
        condition_time=args.cond_time,
        forecast_horizon=args.forc_time,
        num_folds=args.nfolds,
    )


def _load_ode_dataset(args):
    """Load ODE-based synthetic dataset."""
    from tsdm.tasks.ode_models import ODE_DATASET

    return ODE_DATASET(
        dataname=args.dataset,
        normalize_time=True,
        condition_time=args.cond_time,
        forecast_horizon=args.forc_time,
        num_folds=args.nfolds,
        freq=100,
    )


def create_data_loaders(task, args):
    """
    Create training and evaluation data loaders with optimized settings.
    """
    # Determine optimal number of workers based on CPU cores
    # Rule of thumb: 4 workers per GPU, or num_cpus - 1 for CPU training
    num_workers = min(8, os.cpu_count() - 1) if os.cpu_count() else 4

    train_config = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "pin_memory": True,  # Keep True if using GPU
        "num_workers": num_workers,  # Changed from 0
        "collate_fn": data_utils.tsdm_collate,
        "prefetch_factor": 2,  # Prefetch 2 batches per worker
        "persistent_workers": True,  # Keep workers alive between epochs
    }

    eval_config = {
        "batch_size": 128,  # Increased from 64 (evaluation can use larger batches)
        "shuffle": False,
        "drop_last": False,
        "pin_memory": True,
        "num_workers": num_workers,  # Changed from 0
        "collate_fn": data_utils.tsdm_collate_val,
        "prefetch_factor": 2,
        "persistent_workers": True,
    }

    loaders = {
        "train": task.get_dataloader((args.fold, "train"), **train_config),
        "valid": task.get_dataloader((args.fold, "valid"), **eval_config),
        "test": task.get_dataloader((args.fold, "test"), **eval_config),
    }

    return loaders
