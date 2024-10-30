import os
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point
from datetime import datetime
import pytest

from mobiml.datasets import Dataset, TRAJ_ID, TIMESTAMP

from mobiml.samplers import RandomTrajSampler


class TestRandomTrajSampler:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def setup_method(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 0),
                    "timestamp": datetime(2018, 1, 4, 12, 1, 0),
                    "traj_id": 1,
                },
                {
                    "geometry": Point(1, 2),
                    "timestamp": datetime(2018, 1, 4, 12, 2, 0),
                    "traj_id": 1,
                },
                {
                    "geometry": Point(2, 1),
                    "timestamp": datetime(2018, 1, 4, 12, 3, 0),
                    "traj_id": 2,
                },
                {
                    "geometry": Point(3, 3),
                    "timestamp": datetime(2018, 1, 4, 12, 4, 0),
                    "traj_id": 2,
                },
                {
                    "geometry": Point(1, 5),
                    "timestamp": datetime(2018, 1, 4, 12, 5, 0),
                    "traj_id": 3,
                },
                {
                    "geometry": Point(2, 6),
                    "timestamp": datetime(2018, 1, 4, 12, 6, 0),
                    "traj_id": 3,
                },
                {
                    "geometry": Point(1, 7),
                    "timestamp": datetime(2018, 1, 4, 12, 7, 0),
                    "traj_id": 4,
                },
                {
                    "geometry": Point(3, 8),
                    "timestamp": datetime(2018, 1, 4, 12, 8, 0),
                    "traj_id": 4,
                },
                {
                    "geometry": Point(5, 1),
                    "timestamp": datetime(2018, 1, 4, 12, 9, 0),
                    "traj_id": 5,
                },
                {
                    "geometry": Point(6, 3),
                    "timestamp": datetime(2018, 1, 4, 12, 10, 0),
                    "traj_id": 5,
                },
                {
                    "geometry": Point(6, 2),
                    "timestamp": datetime(2018, 1, 4, 12, 11, 0),
                    "traj_id": 6,
                },
                {
                    "geometry": Point(8, 3),
                    "timestamp": datetime(2018, 1, 4, 12, 12, 0),
                    "traj_id": 6,
                },
                {
                    "geometry": Point(6, 7),
                    "timestamp": datetime(2018, 1, 4, 12, 13, 0),
                    "traj_id": 7,
                },
                {
                    "geometry": Point(8, 8),
                    "timestamp": datetime(2018, 1, 4, 12, 14, 0),
                    "traj_id": 7,
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=4326)

    def test_split(self):
        dataset = Dataset(self.gdf)
        sampler = RandomTrajSampler(dataset)
        data = sampler.split(n_cells=2, n_sample=4, random_state=1)
        assert TRAJ_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert len(data.df) == 7 * 2
        assert len(data.df[data.df.split == 2]) == 4 * 2
        split = [2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2]
        assert data.df["split"].tolist() == split

    def test_sample(self):
        dataset = Dataset(self.gdf)
        sampler = RandomTrajSampler(dataset)
        data = sampler.sample(n_cells=2, percent_sample=0.5, random_state=1)
        assert len(data.df) == 4 * 2

    def test_odd_sample(self):
        dataset = Dataset(self.gdf)
        sampler = RandomTrajSampler(dataset)
        data = sampler.sample(n_cells=2, n_sample=5, random_state=1)
        assert len(data.df) == 5 * 2

    def test_sample_too_big(self):
        dataset = Dataset(self.gdf)
        sampler = RandomTrajSampler(dataset)
        with pytest.raises(ValueError) as excinfo:
            sampler.split(n_cells=2, n_sample=9)
        assert str(excinfo.value) == "Sample too big."

    def test_not_enough_samples(self):
        dataset = Dataset(self.gdf)
        sampler = RandomTrajSampler(dataset)
        with pytest.warns(UserWarning, match=r"Not enough points") as w:
            data = sampler.split(n_cells=2, n_sample=7, random_state=1)
        assert len(data.df) == 7 * 2
        split = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        assert data.df["split"].tolist() == split

    def test_empty_cells(self):
        dataset = Dataset(self.gdf)
        sampler = RandomTrajSampler(dataset)
        with pytest.warns(UserWarning, match=r"empty cells") as w:
            data = sampler.split(n_cells=(4, 2), n_sample=4, random_state=1)
        assert len(data.df) == 7 * 2
        assert len(data.df[data.df.split == 2]) == 4 * 2
        split = [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2]
        print(data.df["split"].tolist())
        assert data.df["split"].tolist() == split
