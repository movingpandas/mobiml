import os
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point
from datetime import datetime

from mobiml.datasets import Dataset, TRAJ_ID, MOVER_ID, TIMESTAMP

from mobiml.transforms.temporal_splitter import TemporalSplitter


class TestTemporalSplitter:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def setup_method(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 0),
                    "timestamp": datetime(2018, 1, 1, 12, 0, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                },
                {
                    "geometry": Point(6, 0),
                    "timestamp": datetime(2018, 1, 1, 12, 6, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                },
                {
                    "geometry": Point(6, 6),
                    "timestamp": datetime(2018, 1, 1, 12, 10, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                },
                {
                    "geometry": Point(9, 9),
                    "timestamp": datetime(2018, 1, 1, 12, 15, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=4326)

    def test_split(self):
        dataset = Dataset(self.gdf)
        splitter = TemporalSplitter(dataset)
        assert isinstance(splitter, TemporalSplitter)
        data = splitter.split()
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert len(data.df) == 4
        split_list = [1, 3, 2, 1]
        assert data.df.split.tolist() == split_list
