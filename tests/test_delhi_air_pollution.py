import os
import pytest
from movingpandas import TrajectoryCollection

from mobiml.datasets import DelhiAirPollution


class TestDelhiAirPollution:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def test_data_from_csv(self):
        path = os.path.join(self.test_dir, "data", "test_2021-01-30_all.csv")
        data = DelhiAirPollution(path)
        assert isinstance(data, DelhiAirPollution)
        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)
        assert len(trajs) > 0
        assert len(data.df) == 10
        assert data.df["x"].iloc[0] == pytest.approx(77.228798)
        assert data.df["y"].iloc[0] == pytest.approx(28.579370)
