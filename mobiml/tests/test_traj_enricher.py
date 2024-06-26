import os
import pandas as pd
from geopandas import GeoDataFrame
from datetime import datetime
from movingpandas import TrajectoryCollection
from shapely.geometry import Point

from mobiml.preprocessing.utils import trajectorycollection_to_df

from mobiml.datasets._dataset import Dataset, TRAJ_ID, MOVER_ID

from mobiml.preprocessing.traj_enricher import TrajectoryEnricher


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
                    "txx": datetime(2018, 1, 1, 12, 6, 0),
                    "tid": 1,
                    "mid": "a",
                },
                {
                    "geometry": Point(6, 6),
                    "txx": datetime(2018, 1, 1, 12, 10, 0),
                    "tid": 1,
                    "mid": "a",
                },
                {
                    "geometry": Point(9, 9),
                    "txx": datetime(2018, 1, 1, 12, 15, 0),
                    "tid": 1,
                    "mid": "a",
                },
            ]
        ).set_index("txx")
        self.df = df

    def test_add_speed(self):
        path = os.path.join(self.test_dir, "data/test_traj_enricher.csv")
        #data = Dataset(path)
        #assert isinstance(data, Dataset)
        data = TrajectoryEnricher(path)
        assert isinstance(data, TrajectoryEnricher)
        #trajs = data.to_trajs()
        #trajs = data.add_speed()
        #assert isinstance(trajs, TrajectoryCollection)
        
        path = os.path.join(self.test_dir, "data/test_traj_enricher_result.csv")
        data.df.to_csv(path)
