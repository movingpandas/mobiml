import os
from movingpandas import TrajectoryCollection

from mobiml.datasets.brest_ais import BrestAIS, PreprocessedBrestAIS

from mobiml.datasets._dataset import TRAJ_ID, MOVER_ID, TIMESTAMP, SPEED, DIRECTION


class TestBrestAIS:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def test_data_from_csv(self):
        path = os.path.join(self.test_dir, "data/test_nari_dynamic.csv")
        data = BrestAIS(path)
        assert isinstance(data, BrestAIS)
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert SPEED in data.df.columns
        assert DIRECTION in data.df.columns
        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)
        assert len(data.df) == 10


class TestPreprocessedBrestAIS:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def test_data_from_csv(self):
        path = os.path.join(
            self.test_dir, "data/test_nautilus_trajectories_preprocessed.csv"
        )
        data = PreprocessedBrestAIS(path)
        assert isinstance(data, PreprocessedBrestAIS)
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert SPEED in data.df.columns
        assert DIRECTION in data.df.columns
        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)
        assert len(data.df) == 10
