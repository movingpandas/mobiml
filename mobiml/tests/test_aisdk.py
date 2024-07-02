import os
from movingpandas import TrajectoryCollection

from mobiml.datasets.aisdk import AISDK, PreprocessedAISDK, SHIPTYPE

from mobiml.datasets._dataset import SPEED, TIMESTAMP, TRAJ_ID, MOVER_ID, DIRECTION


class TestAISDK:
    test_dir = os.path.dirname(os.path.realpath(__file__))

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
