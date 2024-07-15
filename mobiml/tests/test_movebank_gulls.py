import os
import geopandas as gpd
from movingpandas import TrajectoryCollection

from mobiml.datasets import MovebankGulls, TRAJ_ID, MOVER_ID, TIMESTAMP


class TestMovebankGulls:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def test_data_from_csv(self):
        path = os.path.join(self.test_dir, "data/test_gulls.csv")
        data = MovebankGulls(path)
        assert isinstance(data, MovebankGulls)
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)
        assert len(data.df) == 10

    def test_drop_extra_cols(self):
        path = os.path.join(self.test_dir, "data/test_gulls.csv")
        data = MovebankGulls(path)
        extra_cols = {
            "individual-taxon-canonical-name",
            "study-name",
            "location-long",
            "location-lat",
            "event-id",
            "visible",
        }
        for col in extra_cols:
            if col in data.df.columns:
                raise Exception("There is an extra column: {col}".format(col=col))
            else:
                pass
