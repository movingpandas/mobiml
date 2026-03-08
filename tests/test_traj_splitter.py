import pandas as pd
from geopandas import GeoDataFrame
from datetime import datetime, timedelta
from shapely.geometry import Point

from mobiml.datasets import Dataset
from mobiml.preprocessing import TrajectorySplitter


class TestTrajectorySplitter:
    def setup_method(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 0),
                    "txx": datetime(2018, 1, 1, 12, 0, 0),
                    "tid": 1,
                    "mid": "a",
                },
                {
                    "geometry": Point(6, 0),
                    "txx": datetime(2018, 1, 1, 13, 0, 0),
                    "tid": 1,
                    "mid": "a",
                },
                {
                    "geometry": Point(6, 6),
                    "txx": datetime(2018, 1, 2, 16, 0, 0),
                    "tid": 1,
                    "mid": "a",
                },
                {
                    "geometry": Point(9, 9),
                    "txx": datetime(2018, 1, 2, 17, 0, 0),
                    "tid": 1,
                    "mid": "a",
                },
                {
                    "geometry": Point(10, 9),
                    "txx": datetime(2018, 1, 2, 20, 0, 0),
                    "tid": 1,
                    "mid": "a",
                },
                {
                    "geometry": Point(12, 12),
                    "txx": datetime(2018, 1, 2, 21, 0, 0),
                    "tid": 1,
                    "mid": "a",
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=31256)

    def test_split_by_observation_gap(self):
        dataset = Dataset(self.gdf, traj_id="tid", mover_id="mid", timestamp="txx")
        data = TrajectorySplitter(dataset).split(observation_gap=timedelta(hours=10))
        assert len(data.to_trajs()) == 2

    def test_split_chained(self):
        dataset = Dataset(self.gdf, traj_id="tid", mover_id="mid", timestamp="txx")
        data = TrajectorySplitter(dataset).split(observation_gap=timedelta(hours=10))
        data = TrajectorySplitter(data).split(observation_gap=timedelta(hours=2))
        assert len(data.to_trajs()) == 3

    def test_split_by_day(self):
        dataset = Dataset(self.gdf, traj_id="tid", mover_id="mid", timestamp="txx")
        data = TrajectorySplitter(dataset).split(temporal_split_mode="day")
        assert len(data.to_trajs()) == 2
