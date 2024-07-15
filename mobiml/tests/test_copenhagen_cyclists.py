import os
import pandas as pd
from movingpandas import TrajectoryCollection

from mobiml.datasets import CopenhagenCyclists, TRAJ_ID, MOVER_ID, TIMESTAMP, COORDS, ROWNUM


class TestCopenhagenCyclists:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def test_data_from_csv(self):
        path = os.path.join(self.test_dir, "data/test_bike.csv")
        df = pd.read_csv(path)
        df = df.drop(["Unnamed: 0"], axis=1)
        df["xs_640x360"] = df["xs_640x360"].apply(
            lambda s: [float(x.strip(" []")) for x in s.split(",")]
        )
        df["ys_640x360"] = df["ys_640x360"].apply(
            lambda s: [float(x.strip(" []")) for x in s.split(",")]
        )
        data = CopenhagenCyclists(df)
        assert isinstance(data, CopenhagenCyclists)
        assert TRAJ_ID in data.df.columns
        assert TIMESTAMP in data.df.columns
        assert COORDS in data.df.columns
        assert ROWNUM in data.df.columns
        assert MOVER_ID not in data.df.columns
        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)
        assert len(data.df) == 4219

    def test_drop_extra_cols(self):
        path = os.path.join(self.test_dir, "data/test_bike.csv")
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
