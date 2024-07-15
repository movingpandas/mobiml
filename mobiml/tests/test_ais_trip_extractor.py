import os
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point
from datetime import datetime, timedelta

from mobiml.datasets import SPEED
from mobiml.transforms.ais_trip_extractor import AISTripExtractor


class TestAISTripExtractor:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def setup_method(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 0),
                    "timestamp": datetime(2018, 1, 1, 12, 0, 0),
                    "traj_id": 1,
                    "speed": 1,
                },
                {
                    "geometry": Point(6, 0),
                    "timestamp": datetime(2018, 1, 1, 12, 6, 0),
                    "traj_id": 1,
                    "speed": 2,
                },
                {
                    "geometry": Point(6, 6),
                    "timestamp": datetime(2018, 1, 1, 12, 14, 0),
                    "traj_id": 1,
                    "speed": 1,
                },
                {
                    "geometry": Point(6, 9),
                    "timestamp": datetime(2018, 1, 1, 12, 20, 0),
                    "traj_id": 1,
                    "speed": 0,
                },
                {
                    "geometry": Point(9, 9),
                    "timestamp": datetime(2018, 1, 1, 12, 30, 0),
                    "traj_id": 1,
                    "speed": 4,
                },
                {
                    "geometry": Point(12, 9),
                    "timestamp": datetime(2018, 1, 1, 12, 30, 5),
                    "traj_id": 1,
                    "speed": 4,
                },
                {
                    "geometry": Point(12, 12),
                    "timestamp": datetime(2018, 1, 1, 12, 36, 0),
                    "traj_id": 1,
                    "speed": 2,
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=4326)

    def test_trip_extractor(self):
        ex = AISTripExtractor(self.gdf)
        assert isinstance(ex, AISTripExtractor)
        ais_trips = ex.get_trips(
            gap_duration=timedelta(minutes=15),
            generalization_tolerance=timedelta(minutes=1),
        )
        assert len(ais_trips.to_point_gdf()) == 5
        assert len(ais_trips) == 2

    def test_gap_duration(self):
        ex = AISTripExtractor(self.gdf)
        assert isinstance(ex, AISTripExtractor)
        ais_trips = ex.get_trips(
            gap_duration=timedelta(minutes=7),
            generalization_tolerance=timedelta(minutes=1),
        )
        assert len(ais_trips.to_point_gdf()) == 4
        assert len(ais_trips) == 2

    def test_generalization_tolerance(self):
        ex = AISTripExtractor(self.gdf)
        assert isinstance(ex, AISTripExtractor)
        ais_trips = ex.get_trips(
            gap_duration=timedelta(minutes=15),
            generalization_tolerance=timedelta(seconds=1),
        )
        assert len(ais_trips.to_point_gdf()) == 6
        assert len(ais_trips) == 2

    def test_no_speed(self):
        data = self.gdf.drop(columns=SPEED)
        assert SPEED not in data.columns
        ex = AISTripExtractor(data)
        ais_trips = ex.get_trips(
            gap_duration=timedelta(minutes=15),
            generalization_tolerance=timedelta(minutes=1),
        )
        assert len(ais_trips.to_point_gdf()) == 6
        assert len(ais_trips) == 1
