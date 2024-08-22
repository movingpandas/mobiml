import os
import pandas as pd
from geopandas import GeoDataFrame
from datetime import datetime
from shapely.geometry import Point

from mobiml.datasets import Dataset, TRAJ_ID, MOVER_ID, TIMESTAMP
from mobiml.preprocessing import TrajectorySubsampler


class TestTrajectorySubsampler:
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
                    "geometry": Point(3, 0),
                    "txx": datetime(2018, 1, 1, 12, 0, 5),
                    "tid": 1,
                    "mid": "a",
                },
                {
                    "geometry": Point(6, 0),
                    "txx": datetime(2018, 1, 1, 12, 0, 12),
                    "tid": 1,
                    "mid": "a",
                },
                {
                    "geometry": Point(6, 6),
                    "txx": datetime(2018, 1, 1, 12, 0, 14),
                    "tid": 2,
                    "mid": "a",
                },
                {
                    "geometry": Point(9, 9),
                    "txx": datetime(2018, 1, 1, 12, 0, 25),
                    "tid": 2,
                    "mid": "a",
                },
                {
                    "geometry": Point(12, 9),
                    "txx": datetime(2018, 1, 1, 12, 0, 27),
                    "tid": 3,
                    "mid": "a",
                },
                {
                    "geometry": Point(12, 12),
                    "txx": datetime(2018, 1, 1, 12, 0, 36),
                    "tid": 3,
                    "mid": "a",
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=31256)

    def test_subsample(self):
        dataset = Dataset(
            self.gdf, traj_id="tid", mover_id="mid", timestamp="txx"
        )
        assert len(dataset.to_trajs()) == 3
        subsampler = TrajectorySubsampler(dataset)
        assert isinstance(subsampler, TrajectorySubsampler)
        data = subsampler.subsample(min_dt_sec=10)
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert len(data.to_trajs()) == 2
        assert len(data.to_trajs().get_trajectory(1).df) == 2

    def test_subsample_trajectory(self):
        dataset = Dataset(
            self.gdf, traj_id="tid", mover_id="mid", timestamp="txx"
        )
        assert len(dataset.to_trajs()) == 3
        df = dataset.to_df()
        subsampler_traj = TrajectorySubsampler(df)
        assert isinstance(subsampler_traj, TrajectorySubsampler)
        data = subsampler_traj._subsample_trajectory(traj_df=df, min_dt_sec=10)
        assert TRAJ_ID in data.columns
        assert MOVER_ID in data.columns
        assert TIMESTAMP in data.columns
        assert len(data) == 4
