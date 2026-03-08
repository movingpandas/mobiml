import os
import pytest
from movingpandas import TrajectoryCollection

from mobiml.datasets import (
    CopenhagenCyclists,
    MOVER_ID,
    COORDS,
    ROWNUM,
)
from mobiml.datasets.utils import get_x_from_xy, get_y_from_xy


class TestCopenhagenCyclists:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def setup_method(self):
        path = os.path.join(self.test_dir, "data", "test_bike.pickle")
        self.data = CopenhagenCyclists(path)

    def test_coordinates_are_within_frame_bounds(self):
        assert isinstance(self.data, CopenhagenCyclists)
        assert len(self.data.df) == 4219
        assert COORDS in self.data.df.columns
        assert ROWNUM in self.data.df.columns
        assert MOVER_ID not in self.data.df.columns

        x = get_x_from_xy(self.data.df)
        y = get_y_from_xy(self.data.df)
        assert x.between(0, 640).all(), "x coordinates out of bounds [0, 640]"
        assert y.between(0, 360).all(), "y coordinates out of bounds [0, 360]"
        assert x.iloc[0] == pytest.approx(483.545)
        assert y.iloc[0] == pytest.approx(181.87)

        trajs = self.data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)
        assert len(trajs) > 0

    def test_drop_extra_cols(self):
        extra_cols = {
            "frame_out",
            "num_frames",
            "time_on_screen_s",
            "x_start_640x360",
            "x_end_640x360",
            "y_start_640x360",
            "y_end_640x360",
            "class",
        }
        for col in extra_cols:
            assert col not in self.data.df.columns
