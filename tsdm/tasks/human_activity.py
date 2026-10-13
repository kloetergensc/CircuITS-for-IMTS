r"""Forecasting task on the UCI Human Activity dataset.

The evaluation protocol follows t-PatchGNN / `PyOmniTS
<https://github.com/Ladbaby/PyOmniTS>`_: each record is cut into windows of 4000ms with a
stride of 4000ms, the first ``condition_time`` milliseconds of a window are observed and
everything that is observed in the remaining ``forecast_horizon`` milliseconds has to be
forecast.

Because the window takes the role of the 48 hours of the clinical tasks, the conditioning
range and the forecast horizon are given on the same scale as everywhere else in this
package, i.e. ``-ct 36 -ft 12`` → 3000ms/1000ms, ``-ct 24 -ft 24`` → 2000ms/2000ms,
``-ct 12 -ft 36`` → 1000ms/3000ms and ``-ct 36 -ft 0`` → 3000ms plus the next 3 observed
timestamps. Values of at least 100 are read as milliseconds instead, which gives access
to the second setting used by PyOmniTS (``-ct 3000 -ft 300``).

The splits are made on the level of *records*, not windows, so no window of a test
record leaks into training. Fold 0 reproduces the split of PyOmniTS exactly
(90%/10% twice, without shuffling → 949/193/218 windows), the remaining folds use the
same proportions with shuffled records.
"""

__all__ = [
    "HumanActivity",
    "Sample",
    "Batch",
    "TaskDataset",
]

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
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tsdm.datasets.human_activity import HumanActivity as HumanActivity_Dataset
from tsdm.encoders import FrameEncoder, Standardizer
from tsdm.tasks.base import BaseTask
from tsdm.utils import is_partition
from tsdm.utils.strings import repr_namedtuple


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
        observations = t <= self.observation_time
        first_target = observations.sum()
        sample_mask = slice(0, first_target)
        target_mask = slice(first_target, first_target + self.prediction_steps)
        return Sample(
            key=key,
            inputs=Inputs(t[sample_mask], x[sample_mask], t[target_mask]),
            targets=x[target_mask],
            originals=(t, x),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class HumanActivity(BaseTask):
    r"""Windowed forecasting task on the UCI Human Activity dataset."""

    RANDOM_STATE = 42
    test_size = 0.1  # of total, as in PyOmniTS
    valid_size = 0.1  # of train, i.e. 0.9*0.1 = 0.09 of total

    STRIDE = 4000
    r"""Distance between the starts of two consecutive windows, in milliseconds.

    A window is `STRIDE` milliseconds long as well, i.e. the windows tile the record.
    """

    HOURS_PER_WINDOW = 48
    r"""One window corresponds to the 48 "hours" of the other tasks in this package."""

    MILLISECONDS_THRESHOLD = 100
    r"""`condition_time`/`forecast_horizon` above this are read as milliseconds."""

    DEFAULT_PREDICTION_STEPS = 3
    r"""Number of forecast steps when `forecast_horizon=0`, as for USHCN/Physionet."""

    encoder: FrameEncoder[Standardizer, None]

    def __init__(
        self,
        normalize_time: bool = True,
        condition_time: int = 36,
        forecast_horizon: int = 0,
        num_folds: int = 5,
        normalize_values: bool = True,
    ):
        super().__init__()
        self.condition_time = self._to_milliseconds(condition_time)
        if not 0 < self.condition_time < self.STRIDE:
            raise ValueError(f"{condition_time=} must be inside the {self.STRIDE}ms window.")

        if forecast_horizon == 0:
            # like USHCN/Physionet2012: forecast the next `DEFAULT_PREDICTION_STEPS`
            # observed timestamps, here searched for in the rest of the window.
            self.forecast_horizon = self.STRIDE - self.condition_time
            self._prediction_steps = self.DEFAULT_PREDICTION_STEPS
        else:
            self.forecast_horizon = self._to_milliseconds(forecast_horizon)
            self._prediction_steps = None

        self.window = self.condition_time + self.forecast_horizon
        self.num_folds = num_folds
        self.normalize_time = normalize_time
        self.normalize_values = normalize_values
        self.encoder = FrameEncoder(column_encoders=Standardizer())

        # Timestamps live on an integer millisecond grid and observations exactly at
        # `condition_time` belong to the forecast window, hence the half-step offset.
        self.observation_time = (self.condition_time - 0.5) / (self.window if normalize_time else 1)
        print(
            f"HumanActivity: conditioning on {self.condition_time}ms,"
            f" forecasting {self.forecast_horizon}ms"
            + (f" (first {self._prediction_steps} steps)" if self._prediction_steps else "")
        )
        self.IDs = self.dataset.reset_index()["ID"].unique()

    @classmethod
    def _to_milliseconds(cls, value: int) -> int:
        r"""Convert a conditioning range / forecast horizon to milliseconds.

        The observation window of this dataset is `STRIDE` milliseconds long, which plays
        the role of the 48 hours of the clinical datasets. Small values are therefore read
        on that scale (``-ct 36 -ft 12`` → 3000ms/1000ms, ``-ct 24 -ft 24`` → 2000ms/2000ms),
        while values of at least `MILLISECONDS_THRESHOLD` are read as milliseconds directly
        (``-ct 3000 -ft 300``, the second setting used by PyOmniTS).
        """
        if value >= cls.MILLISECONDS_THRESHOLD:
            return int(value)
        return round(value * cls.STRIDE / cls.HOURS_PER_WINDOW)

    @cached_property
    def dataset(self) -> DataFrame:
        r"""Load the dataset and cut it into windows.

        The frame is indexed by ``(ID, Time)``, where ``ID`` identifies a window
        (``"<record>_<window>"``) and ``Time`` is the time within the window.
        """
        ts = HumanActivity_Dataset().dataset

        if self.normalize_values:
            # Standardization is performed over the full data slice, including test,
            # in line with the other tasks in this package.
            self.encoder.fit(ts)
            ts = self.encoder.encode(ts)

        windows = []
        for record_id, frame in ts.groupby(level="ID", sort=True):
            time = frame.index.get_level_values("Time").to_numpy()
            t_max = int(time.max())

            for k, start in enumerate(range(0, t_max - self.condition_time, self.STRIDE)):
                end = start + self.window
                # the last window of a record is closed, all others are half-open
                selected = (time >= start) & (time <= end if end >= t_max else time < end)
                window = frame[selected].copy()

                t = (time[selected] - start).astype("float32")
                if self.normalize_time:
                    t = t / self.window

                window.index = MultiIndex.from_arrays(
                    [np.full(len(window), f"{record_id}_{k}"), t], names=["ID", "Time"]
                )
                windows.append(window)

        return pd.concat(windows)

    @cached_property
    def records(self) -> list[str]:
        r"""The record (person + trial) each window was cut from, in file order."""
        return list(dict.fromkeys(self.record_of(_id) for _id in self.IDs))

    @staticmethod
    def record_of(window_id: str) -> str:
        r"""Return the record a window belongs to."""
        return window_id.rsplit("_", 1)[0]

    @cached_property
    def folds(self) -> list[dict[str, Sequence[str]]]:
        r"""Create the folds, splitting on the level of records."""
        folds = []
        for k in range(self.num_folds):
            # fold 0 reproduces the split used by PyOmniTS / t-PatchGNN
            shuffle = k > 0
            random_state = self.RANDOM_STATE + k if shuffle else None
            train_rec, test_rec = train_test_split(
                self.records, test_size=self.test_size, random_state=random_state, shuffle=shuffle
            )
            train_rec, valid_rec = train_test_split(
                train_rec, test_size=self.valid_size, random_state=random_state, shuffle=shuffle
            )
            fold = {
                key: [_id for _id in self.IDs if self.record_of(_id) in set(split)]
                for key, split in (("train", train_rec), ("valid", valid_rec), ("test", test_rec))
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
        r"""Return sparse table with indices for each split.

        Returns
        -------
        DataFrame[bool]
        """
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
        for _id, frame in self.dataset.groupby(level="ID", sort=False):
            t = torch.tensor(frame.index.get_level_values("Time").to_numpy(), dtype=torch.float32)
            x = torch.tensor(frame.to_numpy(), dtype=torch.float32)
            tensors[_id] = (t, x)
        return tensors

    @cached_property
    def prediction_steps(self) -> int:
        r"""Number of target timestamps.

        Either the fixed number of steps requested via ``forecast_horizon=0``, or the
        largest number of observed timestamps falling into the forecast window of any
        window, so that the whole forecast window is used.
        """
        if self._prediction_steps is not None:
            return self._prediction_steps
        return max(int((t > self.observation_time).sum()) for t, _ in self.tensors.values())

    def get_dataloader(self, key: tuple[int, str], /, **dataloader_kwargs: Any) -> DataLoader:
        r"""Return the dataloader for the given key."""
        fold, partition = key
        fold_idx = set(self.folds[fold][partition])
        dataset = TaskDataset(
            [val for idx, val in self.tensors.items() if idx in fold_idx],
            observation_time=self.observation_time,
            prediction_steps=self.prediction_steps,
        )
        kwargs: dict[str, Any] = {"collate_fn": lambda *x: x} | dataloader_kwargs
        return DataLoader(dataset, **kwargs)
