import os
import pytest
from movingpandas import TrajectoryCollection

from mobiml.datasets import MovebankGulls, TRAJ_ID, MOVER_ID, TIMESTAMP


class TestMovebankGulls:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def setup_method(self):
        path = os.path.join(self.test_dir, "data", "test_gulls.csv")
        self.data = MovebankGulls(path)

    def test_data_from_csv(self):
        assert isinstance(self.data, MovebankGulls)
        assert TRAJ_ID in self.data.df.columns
        assert MOVER_ID in self.data.df.columns
        assert TIMESTAMP in self.data.df.columns
        trajs = self.data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)
        assert len(trajs) > 0
        assert len(self.data.df) == 10
        assert self.data.df["x"].iloc[0] == pytest.approx(24.58617)
        assert self.data.df["y"].iloc[0] == pytest.approx(61.24783)

    def test_drop_extra_cols(self):
        extra_cols = {
            "individual-taxon-canonical-name",
            "study-name",
            "location-long",
            "location-lat",
            "event-id",
            "visible",
        }
        for col in extra_cols:
            assert col not in self.data.df.columns
