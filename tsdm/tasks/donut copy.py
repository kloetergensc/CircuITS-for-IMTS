from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Index, MultiIndex
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch import nan as NAN
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from tsdm.encoders import FrameEncoder, MinMaxScaler, Standardizer
from tsdm.tasks.base import BaseTask
from tsdm.utils import is_partition
from tsdm.utils.strings import repr_namedtuple

r"""Synthetic Uniform Noise dataset."""

__all__ = [
    "UniformNoise",
    "uniform_noise_collate",
    "Sample",
    "Batch",
    "TaskDataset",
]


class Inputs(NamedTuple):
    r"""A single sample of the data."""

    t: Tensor
    x: Tensor
    t_target: Tensor

    def __repr__(self) -> str:
        r"""Return string representation."""
        return repr_namedtuple(self, recursive=False)


class Sample(NamedTuple):
    r"""A single sample of the data."""

    key: int
    inputs: Inputs
    targets: Tensor
    originals: tuple[Tensor, Tensor]

    def __repr__(self) -> str:
        r"""Return string representation."""
        return repr_namedtuple(self, recursive=False)


class Batch(NamedTuple):
    r"""A single sample of the data."""

    x_time: Tensor  # B×N:   the input timestamps.
    x_vals: Tensor  # B×N×D: the input values.
    x_mask: Tensor  # B×N×D: the input mask.

    y_time: Tensor  # B×K:   the target timestamps.
    y_vals: Tensor  # B×K×D: the target values.
    y_mask: Tensor  # B×K×D: the target mask.

    def __repr__(self) -> str:
        return repr_namedtuple(self, recursive=False)


@dataclass
class TaskDataset(Dataset):
    r"""Wrapper for creating samples of the dataset."""

    tensors: list[tuple[Tensor, Tensor]]
    observation_time: float
    prediction_steps: int

    def __len__(self) -> int:
        r"""Return the number of samples in the dataset."""
        return len(self.tensors)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        r"""Return an iterator over the dataset."""
        return iter(self.tensors)

    def __getitem__(self, key: int) -> Sample:
        t, x = self.tensors[key]
        sample_mask = slice(0, 1)
        target_mask = slice(1, 2)
        return Sample(
            key=key,
            inputs=Inputs(t[sample_mask], x[sample_mask], t[target_mask]),
            targets=x[target_mask],
            originals=(t, x),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class Donut(BaseTask):
    r"""Synthetic Uniform Noise dataset."""

    RANDOM_STATE = 42
    test_size = 0.15  # of total
    valid_size = 0.2  # of train split size

    def __init__(
        self,
        normalize_time: bool = True,
        condition_time: float = 5.0,
        forecast_horizon: int = 10,
        num_folds: int = 5,
        num_samples: int = 5000,
        num_dimensions: int = 2,
        time_steps: int = 100,
        dt: float = 0.1,
        missing_rate: float = 0.1,
    ):
        super().__init__()
        self.prediction_steps = 2
        self.observation_time = condition_time
        self.num_folds = num_folds
        self.num_samples = num_samples
        self.num_dimensions = num_dimensions
        self.time_steps = time_steps
        self.dt = dt
        self.missing_rate = missing_rate

        self.encoder = FrameEncoder(
            column_encoders=Standardizer(),
            index_encoders={"Time": MinMaxScaler()},
        )
        self.normalize_time = normalize_time
        self.IDs = list(range(num_samples))

    def _generate_donut(self, sample_id: int) -> np.ndarray:
        """Generate uniform noise sampled from [0,1] for one channel."""
        # Set seed for reproducibility per channel and sample
        np.random.seed(sample_id)

        # Generate uniform noise from [0,1]
        r = np.random.beta(10, 3.0)
        theta = 2.0 * np.pi * np.random.uniform(0, 1)
        a = r * np.cos(theta) * 2
        b = r * np.sin(theta) * 2

        return np.array([[0, 0], [a, b]])

    @cached_property
    def dataset(self) -> DataFrame:
        r"""Generate synthetic uniform noise dataset."""
        np.random.seed(self.RANDOM_STATE)

        # Generate time grid

        data_list = []

        for sample_id in range(self.num_samples):
            # Generate uniform noise paths for all channels

            X = self._generate_donut(sample_id)

            # Create DataFrame for this sample
            sample_df = DataFrame(X, index=pd.Index([1, 2], name="Time"), columns=["0", "1"])

            # Add sample ID as index level
            sample_df.index = pd.MultiIndex.from_product([[sample_id], sample_df.index], names=["RecordID", "Time"])

            # Introduce random missingness
            mask = np.zeros(sample_df.shape, dtype=bool)
            sample_df = sample_df.mask(mask)

            data_list.append(sample_df)

        # Concatenate all samples
        ts = pd.concat(data_list)

        # Fit encoders and transform

        return ts

    @cached_property
    def folds(self) -> list[dict[str, Sequence[int]]]:
        r"""Create the folds."""
        folds = []
        for fold_idx in range(self.num_folds):
            # Use different random state for each fold
            train_idx, test_idx = train_test_split(self.IDs, test_size=self.test_size, random_state=self.RANDOM_STATE + fold_idx)
            train_idx, valid_idx = train_test_split(train_idx, test_size=self.valid_size, random_state=self.RANDOM_STATE + fold_idx)
            fold = {
                "train": train_idx,
                "valid": valid_idx,
                "test": test_idx,
            }
            assert is_partition(fold.values(), union=self.IDs)
            folds.append(fold)

        return folds

    @cached_property
    def split_idx(self):
        r"""Create the split index."""
        fold_idx = Index(list(range(len(self.folds))), name="fold")
        splits = DataFrame(index=self.IDs, columns=fold_idx, dtype="string")

        for k in range(self.num_folds):
            for key, split in self.folds[k].items():
                mask = splits.index.isin(split)
                splits[k] = splits[k].where(~mask, key)
        return splits

    @cached_property
    def split_idx_sparse(self) -> DataFrame:
        r"""Return sparse table with indices for each split."""
        df = self.split_idx
        columns = df.columns

        categories = {col: df[col].astype("category").dtype.categories for col in columns}

        if isinstance(df.columns, MultiIndex):
            index_tuples = [(*col, cat) for col, cats in zip(columns, categories) for cat in categories[col]]
            names = df.columns.names + ["partition"]
        else:
            index_tuples = [(col, cat) for col, cats in zip(columns, categories) for cat in categories[col]]
            names = [df.columns.name, "partition"]

        new_columns = MultiIndex.from_tuples(index_tuples, names=names)
        result = DataFrame(index=df.index, columns=new_columns, dtype=bool)

        if isinstance(df.columns, MultiIndex):
            for col in new_columns:
                result[col] = df[col[:-1]] == col[-1]
        else:
            for col in new_columns:
                result[col] = df[col[0]] == col[-1]

        return result

    @cached_property
    def test_metric(self) -> Callable[[Tensor, Tensor], Tensor]:
        r"""The test metric."""
        return nn.MSELoss()

    @cached_property
    def splits(self) -> Mapping:
        r"""Create the splits."""
        splits = {}
        for key in self.index:
            mask = self.split_idx_sparse[key]
            ids = self.split_idx_sparse.index[mask]
            splits[key] = self.dataset.loc[ids]
        return splits

    @cached_property
    def index(self) -> MultiIndex:
        r"""Create the index."""
        return self.split_idx_sparse.columns

    @cached_property
    def tensors(self) -> Mapping:
        r"""Tensor dictionary."""
        tensors = {}
        for _id in self.IDs:
            s = self.dataset.loc[_id]
            t = torch.tensor(s.index.values, dtype=torch.float32)
            x = torch.tensor(s.values, dtype=torch.float32)
            tensors[_id] = (t, x)
        return tensors

    def get_dataloader(self, key: tuple[int, str], /, **dataloader_kwargs: Any) -> DataLoader:
        r"""Return the dataloader for the given key."""
        fold, partition = key
        fold_idx = self.folds[fold][partition]
        dataset = TaskDataset(
            [val for idx, val in self.tensors.items() if idx in fold_idx],
            observation_time=self.observation_time,
            prediction_steps=self.prediction_steps,
        )
        kwargs: dict[str, Any] = {"collate_fn": lambda *x: x} | dataloader_kwargs
        return DataLoader(dataset, **kwargs)
