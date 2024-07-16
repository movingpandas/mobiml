import os
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point
from datetime import datetime, timedelta

from mobiml.datasets._dataset import MOVER_ID

from mobiml.transforms.ais_trip_extractor import AISTripExtractor

from mobiml.transforms.traj_aggregator import TrajectoryAggregator


class TestTrajectoryAggregator:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def setup_method(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 0),
                    "timestamp": datetime(2018, 1, 1, 12, 0, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                    "speed": 1,
                    "direction": 90.0,
                    "ship_type": "Passenger",
                    "Name": "SHIP1",
                    "client": 1,
                },
                {
                    "geometry": Point(3, 0),
                    "timestamp": datetime(2018, 1, 1, 12, 6, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                    "speed": 3,
                    "direction": 180.0,
                    "ship_type": "Passenger",
                    "Name": "SHIP1",
                    "client": 1,
                },
                {
                    "geometry": Point(3, 6),
                    "timestamp": datetime(2018, 1, 1, 12, 20, 0),
                    "traj_id": 2,
                    "mover_id": 2,
                    "speed": 2,
                    "direction": 270.0,
                    "ship_type": "Passenger",
                    "Name": "SHIP2",
                    "client": 1,
                },
                {
                    "geometry": Point(6, 9),
                    "timestamp": datetime(2018, 1, 1, 12, 25, 0),
                    "traj_id": 2,
                    "mover_id": 2,
                    "speed": 4,
                    "direction": 90.0,
                    "ship_type": "Passenger",
                    "Name": "SHIP2",
                    "client": 1,
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=4326)

    def test_aggregated_trajs(self):
        h3_resolution = 2
        vessels = self.gdf.groupby(MOVER_ID)[["ship_type", "Name"]].agg(pd.Series.mode)
        trajs = AISTripExtractor(self.gdf).get_trips(gap_duration=timedelta(minutes=10))
        trajs = TrajectoryAggregator(trajs, vessels).aggregate_trajs(h3_resolution)

        expected_speed_median = [2, 3]
        trajs_speed = trajs.speed_median.tolist()
        assert trajs_speed[0] == expected_speed_median[0]
        assert trajs_speed[1] == expected_speed_median[1]

        expected_direction_start = [90.0, 270.0]
        trajs_direction = trajs.direction_start.tolist()
        assert trajs_direction[0] == expected_direction_start[0]
        assert trajs_direction[1] == expected_direction_start[1]
