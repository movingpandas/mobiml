import os
import pandas as pd
from geopandas import GeoDataFrame
from datetime import datetime
from shapely.geometry import Point

from mobiml.datasets import Dataset, SPEED
from mobiml.preprocessing import TrajectoryFilter, TrajectoryEnricher


class TestTrajectoryFilter:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def setup_method(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 0),
                    "txx": datetime(2018, 1, 1, 12, 0, 0),
                    "tid": 3,
                    "mid": "a",
                    "speed": 1,
                },
                {
                    "geometry": Point(6, 0),
                    "txx": datetime(2018, 1, 1, 12, 0, 1),
                    "tid": 3,
                    "mid": "a",
                    "speed": 4,
                },
                {
                    "geometry": Point(6, 6),
                    "txx": datetime(2018, 1, 1, 12, 0, 2),
                    "tid": 3,
                    "mid": "a",
                    "speed": 7,
                },
                {
                    "geometry": Point(9, 9),
                    "txx": datetime(2018, 1, 1, 12, 0, 3),
                    "tid": 2,
                    "mid": "a",
                    "speed": 6,
                },
                {
                    "geometry": Point(6, 9),
                    "txx": datetime(2018, 1, 1, 12, 0, 4),
                    "tid": 2,
                    "mid": "a",
                    "speed": 9,
                },
                {
                    "geometry": Point(9, 12),
                    "txx": datetime(2018, 1, 1, 12, 0, 20),
                    "tid": 4,
                    "mid": "a",
                    "speed": 11,
                },
                {
                    "geometry": Point(12, 12),
                    "txx": datetime(2018, 1, 1, 12, 0, 21),
                    "tid": 4,
                    "mid": "a",
                    "speed": 12,
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=31256)

    def test_filter_min_pts(self):
        dataset = Dataset(self.gdf, traj_id="tid", mover_id="mid", timestamp="txx")
        filter = TrajectoryFilter(dataset)
        data = filter.filter_min_pts(min_pts=3)
        assert SPEED in data.df.columns
        assert len(data.to_trajs()) == 1

    def test_filter_speed(self):
        dataset = Dataset(self.gdf, traj_id="tid", mover_id="mid", timestamp="txx")
        filter = TrajectoryFilter(dataset)
        data = filter.filter_speed(min_speed=1, max_speed=10)
        assert SPEED in data.df.columns
        assert len(data.to_trajs()) == 2

    def test_filter_speed_with_TrajectoryEnricher(self):
        dataset = Dataset(self.gdf, traj_id="tid", mover_id="mid", timestamp="txx")
        enricher = TrajectoryEnricher(dataset)
        speed = enricher.add_speed(overwrite=True)
        filter = TrajectoryFilter(speed)
        data = filter.filter_speed(min_speed=1, max_speed=5)
        assert SPEED in data.df.columns
        assert len(data.to_trajs()) == 2
