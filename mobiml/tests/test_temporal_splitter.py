import os
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point
from datetime import datetime

from mobiml.datasets import _Dataset, TIMESTAMP, TRAJ_ID
from mobiml.transforms.temporal_splitter import TemporalSplitter


class TestTemporalSplitter:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def setup_method(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 0),
                    "timestamp": datetime(2018, 1, 1, 12, 1, 0),
                    "traj_id": 1,
                },
                {
                    "geometry": Point(1, 1),
                    "timestamp": datetime(2018, 1, 1, 12, 2, 0),
                    "traj_id": 1,
                },
                {
                    "geometry": Point(2, 3),
                    "timestamp": datetime(2018, 1, 2, 12, 3, 0),
                    "traj_id": 2,
                },
                {
                    "geometry": Point(3, 3),
                    "timestamp": datetime(2018, 1, 2, 12, 4, 0),
                    "traj_id": 2,
                },
                {
                    "geometry": Point(4, 5),
                    "timestamp": datetime(2018, 1, 3, 12, 5, 0),
                    "traj_id": 3,
                },
                {
                    "geometry": Point(5, 6),
                    "timestamp": datetime(2018, 1, 3, 12, 6, 0),
                    "traj_id": 3,
                },
                {
                    "geometry": Point(6, 6),
                    "timestamp": datetime(2018, 1, 4, 12, 7, 0),
                    "traj_id": 4,
                },
                {
                    "geometry": Point(6, 7),
                    "timestamp": datetime(2018, 1, 4, 12, 8, 0),
                    "traj_id": 4,
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=4326)

    def test_split(self):
        dataset = _Dataset(self.gdf)
        splitter = TemporalSplitter(dataset)
        assert isinstance(splitter, TemporalSplitter)
        data = splitter.split(dev_size=0.25, test_size=0.25)
        assert TRAJ_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert len(data.df) == 8
        expected = [1, 1, 1, 1, 2, 2, 3, 3]
        result = data.df["split"].tolist()
        print(data.df)
        print(result)
        assert result == expected
