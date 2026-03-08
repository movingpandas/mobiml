import os
import pytest
import pandas as pd
from movingpandas import TrajectoryCollection

from mobiml.datasets import (
    CopenhagenCyclists,
    TRAJ_ID,
    MOVER_ID,
    TIMESTAMP,
    COORDS,
    ROWNUM,
)
from mobiml.datasets.utils import get_x_from_xy, get_y_from_xy


class TestCopenhagenCyclists:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def test_coordinates_are_within_frame_bounds(self):
        path = os.path.join(self.test_dir, "data/test_bike.pickle")
        data = CopenhagenCyclists(path)
        assert isinstance(data, CopenhagenCyclists)

        assert len(data.df) == 4219       
        assert TRAJ_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert COORDS in data.df.columns
        assert ROWNUM in data.df.columns
        assert MOVER_ID not in data.df.columns
        
        x = get_x_from_xy(data.df)
        y = get_y_from_xy(data.df)
        assert x.between(0, 640).all(), "x coordinates out of bounds [0, 640]"
        assert y.between(0, 360).all(), "y coordinates out of bounds [0, 360]"
        assert x.iloc[0] == pytest.approx(483.545)
        assert y.iloc[0] == pytest.approx(181.87)

        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)
        
    def test_drop_extra_cols(self):
        path = os.path.join(self.test_dir, "data/test_bike.pickle")
        data = CopenhagenCyclists(path)
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
            if col in data.df.columns:
                raise Exception("There is an extra column: {col}".format(col=col))
            else:
                pass
