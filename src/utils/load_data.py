import pdb
from typing import NamedTuple

import numpy as np
import properscoring as ps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class Batch(NamedTuple):
    r"""A single sample of the data."""

    xobs: Tensor  # B×N:   the input timestamps.
    mobs: Tensor  # B×N: the input values.

    xq: Tensor  # B×K:   the target time channel queries.
    mq: Tensor  # B×K: the target mask.
    y: Tensor  # B×K: the answers.


class Inputs(NamedTuple):
    r"""A single sample of the data."""

    t: Tensor
    x: Tensor
    t_target: Tensor


class Sample(NamedTuple):
    r"""A single sample of the data."""

    key: int
    inputs: Inputs
    targets: Tensor


def collate_fn(batch: list[Sample]) -> Batch:
    r"""Collate tensors into batch.

    Transform the data slightly: t, x, t_target → T, X where X[t_target:] = NAN
    """
    batch_xobs: list[Tensor] = []
    batch_mobs: list[Tensor] = []
    batch_xq: list[Tensor] = []
    batch_mq: list[Tensor] = []
    batch_y: list[Tensor] = []

    for sample in batch:

        t, x, t_target = sample.inputs
        y = sample.targets

        # create a mask for looking up the target values
        mask_y = y.isfinite().to(x.dtype)
        mask_x = x.isfinite().to(x.dtype)

        mask_x_bool = mask_x.bool()
        mask_y_bool = mask_y.bool()

        # nan to zeros
        xobs = torch.nan_to_num(x)
        xfut = torch.nan_to_num(y)

        nchans = x.shape[-1]
        chans_x = torch.ones_like(x).cumsum(-1) - 1
        time_x = t.unsqueeze(-1).repeat(1, nchans)

        cobs = chans_x[mask_x_bool]
        tobs = time_x[mask_x_bool]
        obs = xobs[mask_x_bool]
        mobs = mask_x[mask_x_bool]

        chans_y = torch.ones_like(y).cumsum(-1) - 1
        time_y = t_target.unsqueeze(-1).repeat(1, nchans)

        cq = chans_y[mask_y_bool]
        tq = time_y[mask_y_bool]
        y = xfut[mask_y_bool]
        mq = mask_y[mask_y_bool]

        xobs = torch.cat((tobs.unsqueeze(-1), cobs.unsqueeze(-1), obs.unsqueeze(-1)), -1)
        xq = torch.cat((tq.unsqueeze(-1), cq.unsqueeze(-1)), -1)

        batch_xobs.append(xobs)
        batch_mobs.append(mobs)

        batch_xq.append(xq)
        batch_mq.append(mq)
        batch_y.append(y)

    return Batch(
        xobs=pad_sequence(batch_xobs, batch_first=True, padding_value=0),
        mobs=pad_sequence(batch_mobs, batch_first=True, padding_value=0),
        xq=pad_sequence(batch_xq, batch_first=True, padding_value=0),
        mq=pad_sequence(batch_mq, batch_first=True, padding_value=0),
        y=pad_sequence(batch_y, batch_first=True, padding_value=0),
    )


def data_loaders(TASK, ARGS, collate_fn=collate_fn) -> tuple:
    r"""make data loaders"""

    dloader_config_train = {
        "batch_size": ARGS.batch_size,
        "shuffle": True,
        "drop_last": True,
        "pin_memory": True,
        "num_workers": 4,
        "collate_fn": collate_fn,
    }

    dloader_config_infer = {
        "batch_size": ARGS.val_batch_size,
        "shuffle": False,
        "drop_last": False,
        "pin_memory": True,
        "num_workers": 4,
        "collate_fn": collate_fn,
    }

    dloader_config_sampling = {
        "batch_size": 1,
        "shuffle": False,
        "drop_last": False,
        "pin_memory": True,
        "num_workers": 4,
        "collate_fn": collate_fn,
    }

    train_loader = TASK.get_dataloader((ARGS.fold, "train"), **dloader_config_train)
    valid_loader = TASK.get_dataloader((ARGS.fold, "valid"), **dloader_config_infer)
    test_loader = TASK.get_dataloader((ARGS.fold, "test"), **dloader_config_infer)
    test_loader_sampling = TASK.get_dataloader((ARGS.fold, "test"), **dloader_config_sampling)

    max_N = 0
    min_N = 1000
    max_Q = 0
    min_Q = 1000
    sum_Q = 0
    sum_N = 0
    n = 0
    all_channel_q = []
    try:
        for loader in [train_loader, valid_loader, test_loader]:
            for batch in loader:
                obs, mobs, xq, mq, y = batch
                all_channel_q += xq[:, :, 1].to(int).flatten()[mq.flatten().to(bool)].tolist()
                n += y.shape[0]
                sum_Q += mq.sum()
                sum_N += mq.sum() + mobs.sum()
                this_min_q = mq.sum(-1).min()
                this_max_q = mq.sum(-1).max()
                this_min_n = (mq.sum(-1) + mobs.sum(-1)).min()
                this_max_n = (mq.sum(-1) + mobs.sum(-1)).max()
                if min_Q > this_min_q:
                    min_Q = this_min_q
                if max_Q < this_max_q:
                    max_Q = this_max_q
                if min_N > this_min_n:
                    min_N = this_min_n
                if max_N < this_max_n:
                    max_N = this_max_n
        mean_Q = sum_Q / n
        mean_N = sum_N / n
        print(f"Sequence lengths - N: [{min_N}, {max_N}] (mean: {mean_N:.2f}), Q: [{min_Q}, {max_Q}] (mean: {mean_Q:.2f}), n_samples: {n}")
        # with open("channel_q_ind.txt", "w") as f:
        #    for s in all_channel_q:
        #        f.write(str(s) + "\n")
    except ValueError:
        print("dataset statistics only works with circuit collate_fn")

    return train_loader, valid_loader, test_loader, test_loader_sampling


def load_dataset(args):
    if args.dataset == "ushcn":
        from tsdm.tasks import USHCN_DeBrouwer2019

        task = USHCN_DeBrouwer2019(
            normalize_time=True,
            condition_time=args.cond_time,
            forecast_horizon=args.forc_time,
            num_folds=args.nfolds,
        )
    elif args.dataset == "mimiciii":
        from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019

        task = MIMIC_III_DeBrouwer2019(
            normalize_time=True,
            condition_time=args.cond_time,
            forecast_horizon=args.forc_time,
            num_folds=args.nfolds,
        )
    elif args.dataset == "mimiciv":
        from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021

        task = MIMIC_IV_Bilos2021(
            normalize_time=True,
            condition_time=args.cond_time,
            forecast_horizon=args.forc_time,
            num_folds=args.nfolds,
        )
    elif args.dataset == "physionet2012":
        from tsdm.tasks.physionet2012 import Physionet2012

        task = Physionet2012(
            normalize_time=True,
            condition_time=args.cond_time,
            forecast_horizon=args.forc_time,
            num_folds=args.nfolds,
        )

    elif args.dataset in ("activity", "human_activity"):
        from tsdm.tasks.human_activity import HumanActivity

        # NOTE: one window is 4000ms and takes the role of the 48 hours of the clinical
        # datasets, i.e. -ct 36 -ft 12 → 3000ms/1000ms. Values >= 100 are read as ms.
        task = HumanActivity(
            normalize_time=True,
            condition_time=args.cond_time,
            forecast_horizon=args.forc_time,
            num_folds=args.nfolds,
        )

    elif args.dataset == "brownian":
        from tsdm.tasks.brownian import BrownianMotion

        task = BrownianMotion(
            normalize_time=True,
            condition_time=args.cond_time,
            forecast_horizon=args.forc_time,
            num_folds=args.nfolds,
            num_dimensions=args.channel,
            time_steps=args.timesteps,
        )

    elif args.dataset == "bifurcation":
        from tsdm.tasks.bifurcation import Bifurcation

        task = Bifurcation(
            normalize_time=True,
            condition_time=args.cond_time,
            forecast_horizon=args.forc_time,
            num_folds=args.nfolds,
        )
    elif args.dataset == "bifurcation_c1_c2":
        from tsdm.tasks.bifurcation_c1c2 import Bifurcation

        task = Bifurcation(
            normalize_time=True,
            condition_time=args.cond_time,
            forecast_horizon=args.forc_time,
            num_folds=args.nfolds,
        )
    elif args.dataset == "bifurcation_c1_c4":
        from tsdm.tasks.bifurcation_c1c4 import Bifurcation

        task = Bifurcation(
            normalize_time=True,
            condition_time=args.cond_time,
            forecast_horizon=args.forc_time,
            num_folds=args.nfolds,
        )
    elif args.dataset == "uniform_noise":
        from tsdm.tasks.uniform_noise import UniformNoise

        task = UniformNoise(
            normalize_time=True,
            condition_time=args.cond_time,
            forecast_horizon=args.forc_time,
            num_folds=args.nfolds,
        )
    elif args.dataset == "donut":
        from tsdm.tasks.donut import Donut

        task = Donut(
            normalize_time=True,
            condition_time=args.cond_time,
            forecast_horizon=args.forc_time,
            num_folds=args.nfolds,
        )
        args.val_batch_size = 1000
        args.plot = True
        from src.utils import plot2D as plot_samples
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return task
