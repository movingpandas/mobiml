import os
from movingpandas import TrajectoryCollection

from mobiml.datasets import DelhiAirPollution, TRAJ_ID, MOVER_ID, TIMESTAMP


class TestDelhiAirPollution:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def test_data_from_csv(self):
        path = os.path.join(self.test_dir, "data/test_2021-01-30_all.csv")
        data = DelhiAirPollution(path)
        assert isinstance(data, DelhiAirPollution)
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)
        assert len(data.df) == 10
