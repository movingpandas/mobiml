import os
import pandas as pd
from geopandas import GeoDataFrame
from datetime import datetime
from shapely.geometry import Point

from mobiml.datasets._dataset import Dataset, TRAJ_ID, MOVER_ID, SPEED, TIMESTAMP

from mobiml.preprocessing.traj_filter import TrajectoryFilter

from mobiml.preprocessing.traj_enricher import TrajectoryEnricher


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
        assert len(dataset.to_trajs()) == 3
        filter = TrajectoryFilter(dataset)
        assert isinstance(filter, TrajectoryFilter)
        data = filter.filter_min_pts(min_pts=3)
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert len(data.to_trajs()) == 1

    def test_filter_speed(self):
        dataset = Dataset(self.gdf, traj_id="tid", mover_id="mid", timestamp="txx")
        assert len(dataset.to_trajs()) == 3
        filter = TrajectoryFilter(dataset)
        assert isinstance(filter, TrajectoryFilter)
        data = filter.filter_speed(min_speed=1, max_speed=10)
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert SPEED in data.df.columns
        assert len(data.to_trajs()) == 2

    def test_filter_speed_with_TrajectoryEnricher(self):
        dataset = Dataset(self.gdf, traj_id="tid", mover_id="mid", timestamp="txx")
        assert len(dataset.to_trajs()) == 3
        enricher = TrajectoryEnricher(dataset)
        assert isinstance(enricher, TrajectoryEnricher)
        speed = enricher.add_speed(overwrite=True)
        filter = TrajectoryFilter(speed)
        assert isinstance(filter, TrajectoryFilter)
        data = filter.filter_speed(min_speed=1, max_speed=5)
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert SPEED in data.df.columns
        assert len(data.to_trajs()) == 2
