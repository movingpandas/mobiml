import os
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point
from datetime import datetime

from mobiml.datasets import Dataset
from mobiml.samplers.temporal_splitter import TemporalSplitter


class TestTemporalSplitter:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    @staticmethod
    def _make_gdf(rows):
        return GeoDataFrame(pd.DataFrame(rows), crs=4326)

    def test_split(self):
        gdf = self._make_gdf([
            {"geometry": Point(0, 0), "timestamp": datetime(2018, 1, 1, 12, 1, 0), "traj_id": 1},
            {"geometry": Point(1, 1), "timestamp": datetime(2018, 1, 1, 12, 2, 0), "traj_id": 1},
            {"geometry": Point(2, 3), "timestamp": datetime(2018, 1, 2, 12, 3, 0), "traj_id": 2},
            {"geometry": Point(3, 3), "timestamp": datetime(2018, 1, 2, 12, 4, 0), "traj_id": 2},
            {"geometry": Point(4, 5), "timestamp": datetime(2018, 1, 3, 12, 5, 0), "traj_id": 3},
            {"geometry": Point(5, 6), "timestamp": datetime(2018, 1, 3, 12, 6, 0), "traj_id": 3},
            {"geometry": Point(6, 6), "timestamp": datetime(2018, 1, 4, 12, 7, 0), "traj_id": 4},
            {"geometry": Point(6, 7), "timestamp": datetime(2018, 1, 4, 12, 8, 0), "traj_id": 4},
        ])
        data = TemporalSplitter(Dataset(gdf)).split(dev_size=0.25, test_size=0.25)
        assert len(data.df) == 8
        assert data.df["split"].tolist() == [1, 1, 1, 1, 2, 2, 3, 3]

    def test_split_hr(self):
        gdf = self._make_gdf([
            {"geometry": Point(0, 0), "timestamp": datetime(2018, 1, 2, 10, 1, 0), "traj_id": 1},
            {"geometry": Point(1, 1), "timestamp": datetime(2018, 1, 2, 10, 2, 0), "traj_id": 1},
            {"geometry": Point(2, 3), "timestamp": datetime(2018, 1, 2, 11, 3, 0), "traj_id": 2},
            {"geometry": Point(3, 3), "timestamp": datetime(2018, 1, 2, 11, 4, 0), "traj_id": 2},
            {"geometry": Point(4, 5), "timestamp": datetime(2018, 1, 2, 12, 5, 0), "traj_id": 3},
            {"geometry": Point(5, 6), "timestamp": datetime(2018, 1, 2, 12, 6, 0), "traj_id": 3},
            {"geometry": Point(6, 6), "timestamp": datetime(2018, 1, 2, 13, 7, 0), "traj_id": 4},
            {"geometry": Point(6, 7), "timestamp": datetime(2018, 1, 2, 13, 8, 0), "traj_id": 4},
        ])
        data = TemporalSplitter(Dataset(gdf)).split_hr(dev_size=0.25, test_size=0.25)
        assert len(data.df) == 8
        assert data.df["split"].tolist() == [1, 1, 1, 1, 2, 2, 3, 3]

    def test_split_at_timestamp(self):
        gdf = self._make_gdf([
            {"geometry": Point(0, 0), "timestamp": datetime(2018, 1, 2, 0, 0, 0), "traj_id": 1},
            {"geometry": Point(1, 1), "timestamp": datetime(2018, 1, 2, 1, 0, 0), "traj_id": 1},
            {"geometry": Point(2, 3), "timestamp": datetime(2018, 1, 2, 2, 0, 0), "traj_id": 2},
            {"geometry": Point(3, 3), "timestamp": datetime(2018, 1, 2, 3, 0, 0), "traj_id": 2},
            {"geometry": Point(4, 5), "timestamp": datetime(2018, 1, 2, 4, 0, 0), "traj_id": 3},
            {"geometry": Point(5, 6), "timestamp": datetime(2018, 1, 2, 5, 0, 0), "traj_id": 3},
        ])
        data = TemporalSplitter(Dataset(gdf)).split_at_timestamp(
            timestamp=datetime(2018, 1, 2, 3, 0, 0)
        )
        assert len(data.df) == 6
        assert data.df["split"].tolist() == [1, 1, 1, 2, 2, 2]

    def test_split_at_two_timestamps(self):
        gdf = self._make_gdf([
            {"geometry": Point(0, 0), "timestamp": datetime(2018, 1, 2, 0, 0, 0), "traj_id": 1},
            {"geometry": Point(1, 1), "timestamp": datetime(2018, 1, 2, 1, 0, 0), "traj_id": 1},
            {"geometry": Point(2, 3), "timestamp": datetime(2018, 1, 2, 2, 0, 0), "traj_id": 2},
            {"geometry": Point(3, 3), "timestamp": datetime(2018, 1, 2, 3, 0, 0), "traj_id": 2},
            {"geometry": Point(4, 5), "timestamp": datetime(2018, 1, 2, 4, 0, 0), "traj_id": 3},
            {"geometry": Point(5, 6), "timestamp": datetime(2018, 1, 2, 5, 0, 0), "traj_id": 3},
            {"geometry": Point(6, 6), "timestamp": datetime(2018, 1, 2, 6, 0, 0), "traj_id": 4},
            {"geometry": Point(6, 7), "timestamp": datetime(2018, 1, 2, 7, 0, 0), "traj_id": 4},
        ])
        data = TemporalSplitter(Dataset(gdf)).split_at_timestamp(
            timestamp=datetime(2018, 1, 2, 2, 0, 0),
            timestamp_2=datetime(2018, 1, 2, 5, 0, 0),
        )
        assert len(data.df) == 8
        assert data.df["split"].tolist() == [1, 1, 2, 2, 2, 3, 3, 3]
