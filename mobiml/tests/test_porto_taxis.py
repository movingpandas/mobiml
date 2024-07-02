import os
from movingpandas import TrajectoryCollection

from mobiml.datasets.porto_taxis import PortoTaxis

from mobiml.datasets._dataset import TRAJ_ID, MOVER_ID, TIMESTAMP, COORDS, ROWNUM


class TestPortoTaxis:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def test_data_from_csv(self):
        path = os.path.join(self.test_dir, "data/test_train.csv")
        data = PortoTaxis(path)
        assert isinstance(data, PortoTaxis)
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert COORDS in data.df.columns
        assert ROWNUM in data.df.columns
        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)
        assert len(data.df) == 332
