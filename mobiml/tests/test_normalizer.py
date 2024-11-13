import os
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point
from datetime import datetime

from mobiml.datasets import Dataset, TRAJ_ID, TIMESTAMP, SPEED, DIRECTION
from mobiml.preprocessing import Normalizer


class TestNormalizer:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def setup_method(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 3),
                    "timestamp": datetime(2018, 1, 1, 12, 0, 0),
                    "traj_id": 1,
                    "speed": 3.0,
                    "direction": 90.0,
                },
                {
                    "geometry": Point(6, 3),
                    "timestamp": datetime(2018, 1, 1, 12, 6, 0),
                    "traj_id": 1,
                    "speed": 2.0,
                    "direction": 180.0,
                },
                {
                    "geometry": Point(6, 6),
                    "timestamp": datetime(2018, 1, 1, 12, 10, 0),
                    "traj_id": 1,
                    "speed": 1.0,
                    "direction": 90.0,
                },
                {
                    "geometry": Point(6, 9),
                    "timestamp": datetime(2018, 1, 1, 12, 15, 0),
                    "traj_id": 1,
                    "speed": 10.0,
                    "direction": 270.0,
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=4326)

    def test_normalize_replace_false(self):
        dataset = Dataset(self.gdf)
        assert isinstance(dataset, Dataset)
        assert len(dataset.df.columns) == 5
        normalizer = Normalizer(dataset)
        data = normalizer.normalize(replace=False)
        assert TRAJ_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert SPEED in data.df.columns
        assert DIRECTION in data.df.columns
        assert len(data.df.columns) == 11
        assert len(data.df) == 4
        norm_x = [0, 1, 1, 1]
        norm_y = [0, 0, 0.5, 1]
        norm_speed = [0.3, 0.2, 0.1, 1]
        norm_direction = [0.25, 0.5, 0.25, 0.75]
        assert data.df["norm_x"].tolist() == norm_x
        assert data.df["norm_y"].tolist() == norm_y
        assert data.df["norm_speed"].tolist() == norm_speed
        assert data.df["norm_direction"].tolist() == norm_direction

    def test_normalize_replace_true(self):
        dataset = Dataset(self.gdf)
        assert isinstance(dataset, Dataset)
        assert len(dataset.df.columns) == 5
        normalizer = Normalizer(dataset)
        data = normalizer.normalize(replace=True)
        assert TRAJ_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert SPEED in data.df.columns
        assert DIRECTION in data.df.columns
        assert len(data.df.columns) == 7
        assert len(data.df) == 4
        x = [0, 1, 1, 1]
        y = [0, 0, 0.5, 1.0]
        speed = [0.3, 0.2, 0.1, 1]
        direction = [0.25, 0.5, 0.25, 0.75]
        assert data.df["x"].tolist() == x
        assert data.df["y"].tolist() == y
        assert data.df["speed"].tolist() == speed
        assert data.df["direction"].tolist() == direction

    def test_max_speed(self):
        dataset = Dataset(self.gdf)
        assert isinstance(dataset, Dataset)
        assert len(dataset.df.columns) == 5
        normalizer = Normalizer(dataset)
        data = normalizer.normalize(speed_max=5.0, replace=True)
        assert TRAJ_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert SPEED in data.df.columns
        assert DIRECTION in data.df.columns
        assert len(data.df.columns) == 7
        assert len(data.df) == 4
        speed = [0.6, 0.4, 0.2, 1]
        assert data.df["speed"].tolist() == speed

    def test_no_speed(self):
        dataset = Dataset(self.gdf)
        assert isinstance(dataset, Dataset)
        assert len(dataset.df.columns) == 5
        dataset.df = dataset.df.drop(columns="speed")
        assert len(dataset.df.columns) == 4
        normalizer = Normalizer(dataset)
        data = normalizer.normalize(replace=True)
        assert TRAJ_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert DIRECTION in data.df.columns
        assert len(data.df) == 4
        assert SPEED not in data.df.columns
        assert len(data.df.columns) == 6

    def test_no_direction(self):
        dataset = Dataset(self.gdf)
        assert isinstance(dataset, Dataset)
        assert len(dataset.df.columns) == 5
        dataset.df = dataset.df.drop(columns="direction")
        assert len(dataset.df.columns) == 4
        normalizer = Normalizer(dataset)
        data = normalizer.normalize(replace=True)
        assert TRAJ_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert SPEED in data.df.columns
        assert len(data.df) == 4
        assert DIRECTION not in data.df.columns
        assert len(data.df.columns) == 6
