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

        xobs = torch.cat(
            (tobs.unsqueeze(-1), cobs.unsqueeze(-1), obs.unsqueeze(-1)), -1
        )
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


def data_loaders(TASK, ARGS) -> tuple:
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

    train_loader = TASK.get_dataloader((ARGS.fold, "train"), **dloader_config_train)
    valid_loader = TASK.get_dataloader((ARGS.fold, "valid"), **dloader_config_infer)
    test_loader = TASK.get_dataloader((ARGS.fold, "test"), **dloader_config_infer)

    return train_loader, valid_loader, test_loader
