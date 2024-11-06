import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from mobiml.datasets import TIMESTAMP


class TemporalSplitter:
    def __init__(self, data: Dataset) -> None:
        self.data = data

    def split(
        self, dev_size=0.2, test_size=0.1, seed=100, stratify=None, **kwargs
    ) -> Dataset:
        """
        Splits dataset temporally into train/dev/test
        (default: 70% train, 20% dev, 10% test)

        This split ensures that the first 70% of days are used to train,
        and the rest are used for dev and test.

        Support for other temporal splits than this date-based split is
        on our todo list.
        """
        self.dev_size = dev_size
        self.test_size = test_size
        self.seed = seed
        self.stratify = stratify

        print(f"{datetime.now()} Splitting dataset ...")

        trajectories_dates = self.data.df[TIMESTAMP].dt.date.sort_values().unique()
        print(trajectories_dates)

        train_indices, dev_indices, test_indices = self._train_test_split(
            trajectories_dates, shuffle=False, **kwargs
        )
        print(f"train: {train_indices}, dev: {dev_indices}, test: {test_indices}")
        train_dates, dev_dates, test_dates = (
            trajectories_dates[train_indices],
            trajectories_dates[dev_indices],
            trajectories_dates[test_indices],
        )

        print(
            f"Train @{(min(train_dates), max(train_dates))=}, "
            + f"\nDev @{(min(dev_dates), max(dev_dates))=}, "
            + f"\nTest @{(min(test_dates), max(test_dates))=}"
        )

        self.data.df.loc[self.data.df[TIMESTAMP].dt.date.isin(train_dates), "split"] = 1
        self.data.df.loc[self.data.df[TIMESTAMP].dt.date.isin(dev_dates), "split"] = 2
        self.data.df.loc[self.data.df[TIMESTAMP].dt.date.isin(test_dates), "split"] = 3

        return self.data

    def split_hr(
        self, dev_size=0.2, test_size=0.1, seed=100, stratify=None, **kwargs
    ) -> Dataset:
        """
        Splits dataset temporally by hours into train/dev/test
        (default: 70% train, 20% dev, 10% test)

        This split ensures that the first 70% of hours are used to train,
        and the rest are used for dev and test.
        """
        self.dev_size = dev_size
        self.test_size = test_size
        self.seed = seed
        self.stratify = stratify

        print(f"{datetime.now()} Splitting dataset by hours ...")

        trajectories_hr = self.data.df[TIMESTAMP].dt.hour.sort_values().unique()
        print(trajectories_hr)

        train_indices, dev_indices, test_indices = self._train_test_split(
            trajectories_hr, shuffle=False, **kwargs
        )
        print(f"train: {train_indices}, dev: {dev_indices}, test: {test_indices}")
        train_hr, dev_hr, test_hr = (
            trajectories_hr[train_indices],
            trajectories_hr[dev_indices],
            trajectories_hr[test_indices],
        )

        print(
            f"Train @{(min(train_hr), max(train_hr))=}, "
            + f"\nDev @{(min(dev_hr), max(dev_hr))=}, "
            + f"\nTest @{(min(test_hr), max(test_hr))=}"
        )

        self.data.df.loc[self.data.df[TIMESTAMP].dt.hour.isin(train_hr), "split"] = 1
        self.data.df.loc[self.data.df[TIMESTAMP].dt.hour.isin(dev_hr), "split"] = 2
        self.data.df.loc[self.data.df[TIMESTAMP].dt.hour.isin(test_hr), "split"] = 3

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

    def split_at_timestamp(self, timestamp=None, timestamp_2=None, **kwargs) -> Dataset:
        """
        Split dataset temporally at timestamp into train/dev,
        and into train/dev/test if two timestamps are provided.
        """
        t_min = self.data.df[TIMESTAMP].min()
        t_max = self.data.df[TIMESTAMP].max()
        t_min = pd.to_datetime(t_min)
        t_max = pd.to_datetime(t_max)
        t_train_max = timestamp - timedelta(seconds=1)

        if timestamp_2 == None:
            t_train_min = pd.to_datetime(t_min)
            t_train_max = pd.to_datetime(t_train_max)
            t_dev_min = pd.to_datetime(timestamp)
            t_dev_max = pd.to_datetime(t_max)
            print(
                "t_train_min:",
                t_train_min,
                "\nt_train_max:",
                t_train_max,
                "\nt_dev_min:",
                t_dev_min,
                "\nt_dev_max:",
                t_dev_max,
            )
            self.data.df = self.data.df[self.data.df[TIMESTAMP] >= t_min]
            self.data.df = self.data.df[self.data.df[TIMESTAMP] <= t_max]
            self.data.df.loc[
                (self.data.df[TIMESTAMP] >= t_train_min)
                & (self.data.df[TIMESTAMP] <= t_train_max),
                "split",
            ] = 1
            self.data.df.loc[
                (self.data.df[TIMESTAMP] >= t_dev_min)
                & (self.data.df[TIMESTAMP] <= t_dev_max),
                "split",
            ] = 2
        else:
            t_dev_max = timestamp_2 - timedelta(seconds=1)
            t_train_min = pd.to_datetime(t_min)
            t_train_max = pd.to_datetime(t_train_max)
            t_dev_min = pd.to_datetime(timestamp)
            t_dev_max = pd.to_datetime(t_dev_max)
            t_test_min = pd.to_datetime(timestamp_2)
            t_test_max = pd.to_datetime(t_max)
            print(
                "t_train_min:",
                t_train_min,
                "\nt_train_max:",
                t_train_max,
                "\nt_dev_min:",
                t_dev_min,
                "\nt_dev_max:",
                t_dev_max,
                "\nt_test_min",
                t_test_min,
                "\nt_test_max",
                t_test_max,
            )
            self.data.df = self.data.df[self.data.df[TIMESTAMP] >= t_min]
            self.data.df = self.data.df[self.data.df[TIMESTAMP] <= t_max]
            self.data.df.loc[
                (self.data.df[TIMESTAMP] >= t_train_min)
                & (self.data.df[TIMESTAMP] <= t_train_max),
                "split",
            ] = 1
            self.data.df.loc[
                (self.data.df[TIMESTAMP] >= t_dev_min)
                & (self.data.df[TIMESTAMP] <= t_dev_max),
                "split",
            ] = 2
            self.data.df.loc[
                (self.data.df[TIMESTAMP] >= t_test_min)
                & (self.data.df[TIMESTAMP] <= t_test_max),
                "split",
            ] = 3

        return self.data
