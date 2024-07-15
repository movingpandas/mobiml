import os
import pandas as pd
from geopandas import GeoDataFrame
from datetime import datetime
from shapely.geometry import Point


from mobiml.datasets import (
    Dataset,
    TRAJ_ID,
    MOVER_ID,
    SPEED,
    DIRECTION,
    TIMESTAMP,
)
from mobiml.preprocessing import TrajectoryEnricher


class TestTrajectoryEnricher:
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
                    "txx": datetime(2018, 1, 1, 12, 0, 1),
                    "tid": 1,
                    "mid": "a",
                },
                {
                    "geometry": Point(6, 6),
                    "txx": datetime(2018, 1, 1, 12, 0, 2),
                    "tid": 1,
                    "mid": "a",
                },
                {
                    "geometry": Point(0, 6),
                    "txx": datetime(2018, 1, 1, 12, 0, 3),
                    "tid": 1,
                    "mid": "a",
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=31256)

    def test_add_speed(self):
        dataset = Dataset(self.gdf, traj_id="tid", mover_id="mid", timestamp="txx")
        enricher = TrajectoryEnricher(dataset)
        assert isinstance(enricher, TrajectoryEnricher)
        data = enricher.add_speed()
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert SPEED in data.df.columns
        speed_list = data.df[SPEED].to_list()
        assert speed_list == [6, 6, 6, 6]

    def test_add_direction(self):
        dataset = Dataset(self.gdf, traj_id="tid", mover_id="mid", timestamp="txx")
        enricher = TrajectoryEnricher(dataset)
        assert isinstance(enricher, TrajectoryEnricher)
        data = enricher.add_direction()
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert DIRECTION in data.df.columns
        direction_list = data.df[DIRECTION].to_list()
        assert direction_list == [90.0, 90.0, 0.0, 270.0]
