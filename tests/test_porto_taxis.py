import os
import pytest
from movingpandas import TrajectoryCollection

from mobiml.datasets import (
    PortoTaxis,
    COORDS,
    ROWNUM,
)
from mobiml.datasets.utils import get_x_from_xy, get_y_from_xy


class TestPortoTaxis:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def test_data_from_csv(self):
        path = os.path.join(self.test_dir, "data", "test_train.csv")
        data = PortoTaxis(path)
        assert isinstance(data, PortoTaxis)
        assert COORDS in data.df.columns
        assert ROWNUM in data.df.columns
        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)
        assert len(trajs) > 0
        assert len(data.df) == 332
        x = get_x_from_xy(data.df)
        y = get_y_from_xy(data.df)
        assert x.iloc[0] == pytest.approx(-8.618643)
        assert y.iloc[0] == pytest.approx(41.141412)
