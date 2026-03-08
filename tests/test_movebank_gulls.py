import os
import pytest
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
        assert len(trajs) > 0
        assert len(data.df) == 10
        assert data.df["x"].iloc[0] == pytest.approx(24.58617)
        assert data.df["y"].iloc[0] == pytest.approx(61.24783)

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
            assert col not in data.df.columns
