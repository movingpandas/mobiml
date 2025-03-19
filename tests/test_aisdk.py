import os
import pandas as pd
from geopandas import GeoDataFrame
from movingpandas import TrajectoryCollection
from shapely.geometry import Point
from datetime import datetime

from mobiml.datasets import (
    AISDK,
    PreprocessedAISDK,
    SHIPTYPE,
    SPEED,
    TIMESTAMP,
    TRAJ_ID,
    MOVER_ID,
    DIRECTION,
)


class TestAISDK:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def setup_method(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 0),
                    "t": datetime(2018, 1, 1, 12, 0, 0),
                    "tid": 1,
                    "mid": "a",
                    "SOG": 15.0,
                    "Heading": 10.0,
                    "Callsign": "SLVG",
                },
                {
                    "geometry": Point(6, 0),
                    "t": datetime(2018, 1, 1, 12, 6, 0),
                    "tid": 1,
                    "mid": "a",
                    "SOG": 25.0,
                    "Heading": 22.0,
                    "Callsign": "SGZO",
                },
                {
                    "geometry": Point(6, 6),
                    "t": datetime(2018, 1, 1, 12, 10, 0),
                    "tid": 1,
                    "mid": "a",
                    "SOG": 10.0,
                    "Heading": 147.0,
                    "Callsign": "H8XS",
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=4326)

    def test_data_from_gdf(self):
        data = AISDK(self.gdf, traj_id="tid", mover_id="mid")
        assert isinstance(data, AISDK)
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert SPEED in data.df.columns
        assert len(data.df.columns) == 5
        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)
        assert len(data.df) == 3

    def test_data_from_csv(self):
        path = os.path.join(self.test_dir, "data/test_aisdk_20180208_sample.csv")
        data = AISDK(path)
        assert isinstance(data, AISDK)
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert SPEED in data.df.columns
        assert DIRECTION in data.df.columns
        assert SHIPTYPE in data.df.columns
        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)
        assert len(data.df) == 4


class TestPreprocessedAISDK:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def test_data_from_feather(self):
        path = os.path.join(self.test_dir, "data/test_ais-extracted-stationary.feather")
        data = PreprocessedAISDK(path)
        assert isinstance(data, PreprocessedAISDK)
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert SPEED in data.df.columns
        assert DIRECTION in data.df.columns
        assert SHIPTYPE in data.df.columns
        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)
        assert len(data.df) == 309813
