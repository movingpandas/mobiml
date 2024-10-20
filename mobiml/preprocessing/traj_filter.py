from mobiml.datasets import _Dataset, SPEED
from tqdm.auto import tqdm

from mobiml.datasets.utils import TRAJ_ID


class TrajectoryFilter:
    def __init__(self, data: _Dataset) -> None:
        self.data = data

    def filter_min_pts(self, min_pts=10) -> _Dataset:
        tqdm.pandas()
        vessels_points = self.data.df[TRAJ_ID].value_counts()

        filtered = self.data.df.loc[
            self.data.df[TRAJ_ID].isin(
                vessels_points.loc[vessels_points >= min_pts].index
            )
        ].copy()
        self.data.df = filtered
        return self.data

    def filter_speed(self, min_speed=None, max_speed=None) -> _Dataset:
        filtered = self.data.df.drop(
            self.data.df.loc[
                ~self.data.df[SPEED].between(min_speed, max_speed, inclusive="both")
            ].index,
            axis=0,
        ).copy()
        self.data.df = filtered
        return self.data
