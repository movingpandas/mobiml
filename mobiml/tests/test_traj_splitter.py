import os
import pandas as pd
from geopandas import GeoDataFrame
from datetime import datetime, timedelta
from shapely.geometry import Point

from mobiml.datasets import Dataset, TRAJ_ID, MOVER_ID, TIMESTAMP
from mobiml.preprocessing import TrajectorySplitter


class TestTrajectorySplitter:
    test_dir = os.path.dirname(os.path.realpath(__file__))

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

    def test_split(self):
        dataset = Dataset(
            self.gdf, traj_id="tid", mover_id="mid", timestamp="txx"
        )
        splitter = TrajectorySplitter(dataset)
        assert isinstance(splitter, TrajectorySplitter)
        data = splitter.split(observation_gap=timedelta(hours=10))
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        trajs = data.to_trajs()
        assert len(trajs) == 2
        splitter = TrajectorySplitter(data)
        data = splitter.split(observation_gap=timedelta(hours=2))
        trajs = data.to_trajs()
        assert len(trajs) == 3
