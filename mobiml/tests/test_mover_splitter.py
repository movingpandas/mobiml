import os
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point
from datetime import datetime

from mobiml.datasets import TRAJ_ID, MOVER_ID, SHIPTYPE

from mobiml.transforms.mover_splitter import MoverSplitter


class TestMoverSplitter:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def setup_method(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 0),
                    "timestamp": datetime(2018, 1, 1, 12, 1, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                    "ship_type": "Passenger",
                },
                {
                    "geometry": Point(1, 1),
                    "timestamp": datetime(2018, 1, 1, 12, 2, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                    "ship_type": "Passenger",
                },
                {
                    "geometry": Point(2, 3),
                    "timestamp": datetime(2018, 1, 1, 12, 3, 0),
                    "traj_id": 2,
                    "mover_id": 2,
                    "ship_type": "Passenger",
                },
                {
                    "geometry": Point(3, 3),
                    "timestamp": datetime(2018, 1, 1, 12, 4, 0),
                    "traj_id": 2,
                    "mover_id": 2,
                    "ship_type": "Passenger",
                },
                {
                    "geometry": Point(4, 5),
                    "timestamp": datetime(2018, 1, 1, 12, 5, 0),
                    "traj_id": 3,
                    "mover_id": 3,
                    "ship_type": "Passenger",
                },
                {
                    "geometry": Point(5, 6),
                    "timestamp": datetime(2018, 1, 1, 12, 6, 0),
                    "traj_id": 3,
                    "mover_id": 3,
                    "ship_type": "Passenger",
                },
                {
                    "geometry": Point(6, 6),
                    "timestamp": datetime(2018, 1, 1, 12, 7, 0),
                    "traj_id": 4,
                    "mover_id": 4,
                    "ship_type": "Passenger",
                },
                {
                    "geometry": Point(6, 7),
                    "timestamp": datetime(2018, 1, 1, 12, 8, 0),
                    "traj_id": 4,
                    "mover_id": 4,
                    "ship_type": "Passenger",
                },
                {
                    "geometry": Point(1, 2),
                    "timestamp": datetime(2018, 1, 1, 12, 9, 0),
                    "traj_id": 5,
                    "mover_id": 5,
                    "ship_type": "Cargo",
                },
                {
                    "geometry": Point(3, 4),
                    "timestamp": datetime(2018, 1, 1, 12, 10, 0),
                    "traj_id": 5,
                    "mover_id": 5,
                    "ship_type": "Cargo",
                },
                {
                    "geometry": Point(5, 5),
                    "timestamp": datetime(2018, 1, 1, 12, 11, 0),
                    "traj_id": 6,
                    "mover_id": 6,
                    "ship_type": "Cargo",
                },
                {
                    "geometry": Point(7, 6),
                    "timestamp": datetime(2018, 1, 1, 12, 12, 0),
                    "traj_id": 6,
                    "mover_id": 6,
                    "ship_type": "Cargo",
                },
                {
                    "geometry": Point(9, 7),
                    "timestamp": datetime(2018, 1, 1, 12, 13, 0),
                    "traj_id": 7,
                    "mover_id": 7,
                    "ship_type": "Cargo",
                },
                {
                    "geometry": Point(10, 8),
                    "timestamp": datetime(2018, 1, 1, 12, 14, 0),
                    "traj_id": 7,
                    "mover_id": 7,
                    "ship_type": "Cargo",
                },
                {
                    "geometry": Point(11, 9),
                    "timestamp": datetime(2018, 1, 1, 12, 15, 0),
                    "traj_id": 8,
                    "mover_id": 8,
                    "ship_type": "Cargo",
                },
                {
                    "geometry": Point(12, 12),
                    "timestamp": datetime(2018, 1, 1, 12, 16, 0),
                    "traj_id": 8,
                    "mover_id": 8,
                    "ship_type": "Cargo",
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=4326)

    def test_split(self):
        trajs = self.gdf
        splitter = MoverSplitter(trajs, mover_id="mover_id", mover_class="ship_type")
        assert isinstance(splitter, MoverSplitter)

        label_col = SHIPTYPE
        features = ["geometry", "traj_id", "mover_id"]
        test_size = 0.25

        data = splitter.split(test_size, features, label_col)
        assert len(data) == 4

        training_features = data[0]
        assert len(training_features) == 12
        assert "geometry" in training_features.columns
        assert TRAJ_ID in training_features.columns
        assert MOVER_ID in training_features.columns

        testing_features = data[1]
        assert len(testing_features) == 4
        assert "geometry" in training_features.columns
        assert TRAJ_ID in training_features.columns
        assert MOVER_ID in training_features.columns

        training_labels = data[2]
        assert len(training_labels) == 12
        assert "Passenger" in training_labels.values
        assert "Cargo" in training_labels.values

        testing_labels = data[3]
        assert len(testing_labels) == 4
        assert "Passenger" in testing_labels.values
        assert "Cargo" in testing_labels.values
