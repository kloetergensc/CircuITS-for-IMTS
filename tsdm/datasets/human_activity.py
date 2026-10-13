r"""UCI "Localization Data for Person Activity" dataset (a.k.a. Human Activity).

Five people wore four tags (left ankle, right ankle, chest, belt) while performing a
scripted sequence of activities. Every tag reports its ``(x, y, z)`` position, which
gives 12 channels in total. Only a single tag is read out at a time, so the resulting
multivariate series is sporadically observed.

The preprocessing follows the one used by Latent-ODE (Rubanova et al.), t-PatchGNN and
`PyOmniTS <https://github.com/Ladbaby/PyOmniTS>`_: the timestamps are quantized to
milliseconds relative to the first observation of a record, and repeated readings of the
same tag within the same millisecond are averaged.
"""

__all__ = [
    # Classes
    "HumanActivity",
]

import logging
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from tsdm.datasets.base import SingleFrameDataset

__logger__ = logging.getLogger(__name__)


class HumanActivity(SingleFrameDataset):
    r"""Localization Data for Person Activity.

    References
    ----------
    - | `Localization Data for Person Activity
        <https://archive.ics.uci.edu/dataset/196/localization+data+for+person+activity>`_
      | Kaluza, Boštjan and Mirchevska, Violeta and Dovgan, Erik and Luštrek, Mitja and Gams, Matjaž
      | UCI Machine Learning Repository, 2010
    - | `Latent ODEs for Irregularly-Sampled Time Series
        <https://proceedings.neurips.cc/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html>`_
      | Rubanova, Yulia and Chen, Ricky T. Q. and Duvenaud, David
    """

    BASE_URL = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00196/"
    r"""HTTP address from where the dataset can be downloaded."""

    INFO_URL = r"https://archive.ics.uci.edu/dataset/196/localization+data+for+person+activity"
    r"""HTTP address containing additional information about the dataset."""

    RAWDATA_SHA256 = "1fbc965b8d7faf499ea854428b2138132d94e6a26404bc8e4b0c62144f513e8a"
    RAWDATA_SHAPE = (164860, 8)
    DATASET_SHAPE = (164834, 12)

    TAG_IDS = [
        "010-000-024-033",  # ANKLE_LEFT
        "010-000-030-096",  # ANKLE_RIGHT
        "020-000-033-111",  # CHEST
        "020-000-032-221",  # BELT
    ]
    r"""Ids of the four tags, in the order used by Rubanova et al."""

    TAG_NAMES = ["ANKLE_LEFT", "ANKLE_RIGHT", "CHEST", "BELT"]
    r"""Human readable names of the four tags."""

    COORDINATES = ["x", "y", "z"]
    r"""The three coordinates reported by each tag."""

    TIME_UNIT = 10**4
    r"""Number of raw ticks (100ns) per millisecond, the resolution used for quantization."""

    rawdata_files = "ConfLongDemo_JSI.txt"
    rawdata_paths: Path

    @property
    def channels(self) -> list[str]:
        r"""Names of the 12 channels, tag-major (as in the original flattening)."""
        return [f"{tag}_{coord}" for tag in self.TAG_NAMES for coord in self.COORDINATES]

    def _clean(self) -> DataFrame:
        r"""Clean an already downloaded raw dataset and store it as a sparse frame."""
        df = pd.read_csv(
            self.rawdata_paths,
            header=None,
            names=["ID", "tag", "time", "date", "x", "y", "z", "label"],
            dtype={
                "ID": "string",
                "tag": "string",
                "time": "int64",
                "date": "string",
                "x": "float64",
                "y": "float64",
                "z": "float64",
                "label": "string",
            },
        )

        if df.shape != self.RAWDATA_SHAPE:
            raise ValueError(f"The {df.shape=} is not correct, expected {self.RAWDATA_SHAPE}.")

        # timestamps are given in 100ns ticks → quantize to milliseconds since the
        # first observation of the respective record.
        first_tick = df.groupby("ID")["time"].transform("first")
        df["Time"] = ((df["time"] - first_tick) / self.TIME_UNIT).round().astype("int32")

        # `reduce="average"`: several readings of one tag within the same millisecond
        # are averaged into a single observation.
        df["tag"] = pd.Categorical(df["tag"].astype(str), categories=self.TAG_IDS, ordered=True)
        values = df.groupby(["ID", "Time", "tag"], observed=True)[self.COORDINATES].mean()

        # → one row per (record, millisecond), NaN where a tag was not read out.
        frame = values.unstack("tag")
        frame.columns = [f"{self.TAG_NAMES[self.TAG_IDS.index(tag)]}_{coord}" for coord, tag in frame.columns]
        frame = frame[self.channels].sort_index().astype("float32")

        if frame.shape != self.DATASET_SHAPE:
            __logger__.warning("The %s is not the expected %s.", frame.shape, self.DATASET_SHAPE)

        return frame
