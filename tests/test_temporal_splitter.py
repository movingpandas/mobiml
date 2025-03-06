import os
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point
from datetime import datetime

from mobiml.datasets import Dataset, TIMESTAMP, TRAJ_ID
from mobiml.samplers.temporal_splitter import TemporalSplitter


class TestTemporalSplitter:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def test_split(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 0),
                    "timestamp": datetime(2018, 1, 1, 12, 1, 0),
                    "traj_id": 1,
                },
                {
                    "geometry": Point(1, 1),
                    "timestamp": datetime(2018, 1, 1, 12, 2, 0),
                    "traj_id": 1,
                },
                {
                    "geometry": Point(2, 3),
                    "timestamp": datetime(2018, 1, 2, 12, 3, 0),
                    "traj_id": 2,
                },
                {
                    "geometry": Point(3, 3),
                    "timestamp": datetime(2018, 1, 2, 12, 4, 0),
                    "traj_id": 2,
                },
                {
                    "geometry": Point(4, 5),
                    "timestamp": datetime(2018, 1, 3, 12, 5, 0),
                    "traj_id": 3,
                },
                {
                    "geometry": Point(5, 6),
                    "timestamp": datetime(2018, 1, 3, 12, 6, 0),
                    "traj_id": 3,
                },
                {
                    "geometry": Point(6, 6),
                    "timestamp": datetime(2018, 1, 4, 12, 7, 0),
                    "traj_id": 4,
                },
                {
                    "geometry": Point(6, 7),
                    "timestamp": datetime(2018, 1, 4, 12, 8, 0),
                    "traj_id": 4,
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=4326)

        dataset = Dataset(self.gdf)
        splitter = TemporalSplitter(dataset)
        assert isinstance(splitter, TemporalSplitter)
        data = splitter.split(dev_size=0.25, test_size=0.25)
        assert TRAJ_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert len(data.df) == 8
        expected = [1, 1, 1, 1, 2, 2, 3, 3]
        result = data.df["split"].tolist()
        assert result == expected

    def test_split_hr(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 0),
                    "timestamp": datetime(2018, 1, 2, 10, 1, 0),
                    "traj_id": 1,
                },
                {
                    "geometry": Point(1, 1),
                    "timestamp": datetime(2018, 1, 2, 10, 2, 0),
                    "traj_id": 1,
                },
                {
                    "geometry": Point(2, 3),
                    "timestamp": datetime(2018, 1, 2, 11, 3, 0),
                    "traj_id": 2,
                },
                {
                    "geometry": Point(3, 3),
                    "timestamp": datetime(2018, 1, 2, 11, 4, 0),
                    "traj_id": 2,
                },
                {
                    "geometry": Point(4, 5),
                    "timestamp": datetime(2018, 1, 2, 12, 5, 0),
                    "traj_id": 3,
                },
                {
                    "geometry": Point(5, 6),
                    "timestamp": datetime(2018, 1, 2, 12, 6, 0),
                    "traj_id": 3,
                },
                {
                    "geometry": Point(6, 6),
                    "timestamp": datetime(2018, 1, 2, 13, 7, 0),
                    "traj_id": 4,
                },
                {
                    "geometry": Point(6, 7),
                    "timestamp": datetime(2018, 1, 2, 13, 8, 0),
                    "traj_id": 4,
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=4326)

        dataset = Dataset(self.gdf)
        splitter = TemporalSplitter(dataset)
        assert isinstance(splitter, TemporalSplitter)
        data = splitter.split_hr(dev_size=0.25, test_size=0.25)
        assert TRAJ_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert len(data.df) == 8
        expected = [1, 1, 1, 1, 2, 2, 3, 3]
        result = data.df["split"].tolist()
        assert result == expected

    def test_split_at_timestamp(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 0),
                    "timestamp": datetime(2018, 1, 2, 0, 0, 0),
                    "traj_id": 1,
                },
                {
                    "geometry": Point(1, 1),
                    "timestamp": datetime(2018, 1, 2, 1, 0, 0),
                    "traj_id": 1,
                },
                {
                    "geometry": Point(2, 3),
                    "timestamp": datetime(2018, 1, 2, 2, 0, 0),
                    "traj_id": 2,
                },
                {
                    "geometry": Point(3, 3),
                    "timestamp": datetime(2018, 1, 2, 3, 0, 0),
                    "traj_id": 2,
                },
                {
                    "geometry": Point(4, 5),
                    "timestamp": datetime(2018, 1, 2, 4, 0, 0),
                    "traj_id": 3,
                },
                {
                    "geometry": Point(5, 6),
                    "timestamp": datetime(2018, 1, 2, 5, 0, 0),
                    "traj_id": 3,
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=4326)

        dataset = Dataset(self.gdf)
        splitter = TemporalSplitter(dataset)
        assert isinstance(splitter, TemporalSplitter)
        data = splitter.split_at_timestamp(timestamp=datetime(2018, 1, 2, 3, 0, 0))
        assert TRAJ_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert len(data.df) == 6
        expected = [1, 1, 1, 2, 2, 2]
        result = data.df["split"].tolist()
        assert result == expected

    def test_split_at_timestamp_2(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 0),
                    "timestamp": datetime(2018, 1, 2, 0, 0, 0),
                    "traj_id": 1,
                },
                {
                    "geometry": Point(1, 1),
                    "timestamp": datetime(2018, 1, 2, 1, 0, 0),
                    "traj_id": 1,
                },
                {
                    "geometry": Point(2, 3),
                    "timestamp": datetime(2018, 1, 2, 2, 0, 0),
                    "traj_id": 2,
                },
                {
                    "geometry": Point(3, 3),
                    "timestamp": datetime(2018, 1, 2, 3, 0, 0),
                    "traj_id": 2,
                },
                {
                    "geometry": Point(4, 5),
                    "timestamp": datetime(2018, 1, 2, 4, 0, 0),
                    "traj_id": 3,
                },
                {
                    "geometry": Point(5, 6),
                    "timestamp": datetime(2018, 1, 2, 5, 0, 0),
                    "traj_id": 3,
                },
                {
                    "geometry": Point(6, 6),
                    "timestamp": datetime(2018, 1, 2, 6, 0, 0),
                    "traj_id": 4,
                },
                {
                    "geometry": Point(6, 7),
                    "timestamp": datetime(2018, 1, 2, 7, 0, 0),
                    "traj_id": 4,
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=4326)

        dataset = Dataset(self.gdf)
        splitter = TemporalSplitter(dataset)
        assert isinstance(splitter, TemporalSplitter)
        data = splitter.split_at_timestamp(
            timestamp=datetime(2018, 1, 2, 2, 0, 0),
            timestamp_2=datetime(2018, 1, 2, 5, 0, 0),
        )
        assert TRAJ_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert len(data.df) == 8
        expected = [1, 1, 2, 2, 2, 3, 3, 3]
        result = data.df["split"].tolist()
        assert result == expected
