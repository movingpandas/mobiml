import os
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point
from datetime import datetime, timedelta

from mobiml.datasets import Dataset, MovebankGulls, SPEED
from mobiml.transforms.traj_creator import TrajectoryCreator


class TestTripExtractor:
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

    def test_input_gdf(self):
        ex = TrajectoryCreator(self.gdf)
        assert isinstance(ex, TrajectoryCreator)
        ais_trips = ex.get_trajs(
            gap_duration=timedelta(minutes=15),
            generalization_tolerance=timedelta(minutes=1),
        )
        assert len(ais_trips.to_point_gdf()) == 5
        assert len(ais_trips) == 2

    def test_gap_duration(self):
        ex = TrajectoryCreator(self.gdf)
        assert isinstance(ex, TrajectoryCreator)
        ais_trips = ex.get_trajs(
            gap_duration=timedelta(minutes=7),
            generalization_tolerance=timedelta(minutes=1),
        )
        assert len(ais_trips.to_point_gdf()) == 4
        assert len(ais_trips) == 2

    def test_generalization_tolerance(self):
        ex = TrajectoryCreator(self.gdf)
        assert isinstance(ex, TrajectoryCreator)
        ais_trips = ex.get_trajs(
            gap_duration=timedelta(minutes=15),
            generalization_tolerance=timedelta(seconds=1),
        )
        assert len(ais_trips.to_point_gdf()) == 6
        assert len(ais_trips) == 2

    def test_no_speed(self):
        data = self.gdf.drop(columns=SPEED)
        assert SPEED not in data.columns
        ex = TrajectoryCreator(data)
        assert isinstance(ex, TrajectoryCreator)
        ais_trips = ex.get_trajs(
            gap_duration=timedelta(minutes=15),
            generalization_tolerance=timedelta(minutes=1),
        )
        assert len(ais_trips.to_point_gdf()) == 6
        assert len(ais_trips) == 1

    def test_input_dataset(self):
        path = os.path.join(self.test_dir, "data/test.csv")
        data = Dataset(
            path,
            name="test",
            traj_id="tid",
            mover_id="mid",
            timestamp="t",
            crs=31256,
        )
        ex = TrajectoryCreator(data)
        assert isinstance(ex, TrajectoryCreator)
        ais_trips = ex.get_trajs(
            gap_duration=timedelta(minutes=15),
            generalization_tolerance=timedelta(minutes=1),
        )
        print(ais_trips.to_traj_gdf())
        assert len(ais_trips.to_point_gdf()) == 4
        assert len(ais_trips) == 1

    def test_input_trajectory_collection(self):
        path = os.path.join(self.test_dir, "data/test_gulls.csv")
        data = MovebankGulls(path)
        trajs = data.to_trajs()
        ex = TrajectoryCreator(trajs)
        assert isinstance(ex, TrajectoryCreator)
        ais_trips = ex.get_trajs(
            gap_duration=timedelta(hours=7),
            generalization_tolerance=timedelta(minutes=1),
        )
        assert len(ais_trips.to_point_gdf()) == 10
        assert len(ais_trips) == 3
