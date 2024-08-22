import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

import pandas as pd
import numpy as np
import pickle, tqdm, os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# import pdb

from mobiml.datasets import TIMESTAMP


class TemporalSplitter:
    def __init__(self, data: Dataset) -> None:
        self.data = data

    def split(
        self, dev_size=0.2, test_size=0.1, seed=100, stratify=None, **kwargs
    ) -> Dataset:
        """
        Splits dataset temporally into train/dev/test (default: 70% train, 20% dev, 10% test)

        This split ensures that the first 70% of days are used to train, and the rest are used for dev and test.

        Support for other temporal splits than this date-based split is on our todo list.
        """
        self.dev_size = dev_size
        self.test_size = test_size
        self.seed = seed
        self.stratify = stratify

        print(f"{datetime.now()} Splitting dataset ...")

        trajectories_dates = (
            self.data.df[TIMESTAMP].dt.date.sort_values().unique()
        )
        print(trajectories_dates)

        train_indices, dev_indices, test_indices = self._train_test_split(
            trajectories_dates, shuffle=False, **kwargs
        )
        print(f"train:{train_indices}; dev:{dev_indices}; test:{test_indices}")
        train_dates, dev_dates, test_dates = (
            trajectories_dates[train_indices],
            trajectories_dates[dev_indices],
            trajectories_dates[test_indices],
        )

        print(
            f"Train @{(min(train_dates), max(train_dates))=};"
            + f"\nDev @{(min(dev_dates), max(dev_dates))=};"
            + f"\nTest @{(min(test_dates), max(test_dates))=}"
        )

        self.data.df.loc[
            self.data.df[TIMESTAMP].dt.date.isin(train_dates), "split"
        ] = 1
        self.data.df.loc[
            self.data.df[TIMESTAMP].dt.date.isin(dev_dates), "split"
        ] = 2
        self.data.df.loc[
            self.data.df[TIMESTAMP].dt.date.isin(test_dates), "split"
        ] = 3

        return self.data

    def _train_test_split(self, dataset, **kwargs):
        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))

        dev_split = self.dev_size + self.test_size
        test_split = self.test_size / dev_split

        train_indices, dev_indices = train_test_split(
            indices,
            test_size=dev_split,
            random_state=self.seed,
            stratify=dataset.labels if self.stratify else None,
            **kwargs,
        )
        dev_indices, test_indices = train_test_split(
            dev_indices,
            test_size=test_split,
            random_state=self.seed,
            stratify=dataset.labels[dev_indices] if self.stratify else None,
            **kwargs,
        )

        return train_indices, dev_indices, test_indices
