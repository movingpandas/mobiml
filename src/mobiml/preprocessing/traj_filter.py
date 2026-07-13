from mobiml.datasets import Dataset, SPEED
from tqdm.auto import tqdm

from mobiml.datasets.utils import TRAJ_ID
from .utils import trajectorycollection_to_df


class TrajectoryFilter:
    def __init__(self, data: Dataset) -> None:
        self.data = data

    def filter_min_pts(self, min_pts=10) -> Dataset:
        tqdm.pandas()
        vessels_points = self.data.df[TRAJ_ID].value_counts()

        filtered = self.data.df.loc[
            self.data.df[TRAJ_ID].isin(
                vessels_points.loc[vessels_points >= min_pts].index
            )
        ].copy()
        self.data.df = filtered
        return self.data

    def filter_kalman(self, process_noise_std=0.5, measurement_noise_std=1) -> Dataset:
        try:
            from movingpandas import KalmanSmootherCV
        except ImportError as error:
            raise ImportError(
                "Missing optional dependency. To use filter_kalman please install "
                "Stone Soup (see "
                "https://stonesoup.readthedocs.io/en/latest/#installation)."
            ) from error

        trajs = self.data.to_trajs()
        smoothed = KalmanSmootherCV(trajs).smooth(
            process_noise_std=process_noise_std,
            measurement_noise_std=measurement_noise_std,
        )
        self.data.df = trajectorycollection_to_df(smoothed)
        return self.data

    def filter_speed(self, min_speed=None, max_speed=None) -> Dataset:
        filtered = self.data.df.drop(
            self.data.df.loc[
                ~self.data.df[SPEED].between(min_speed, max_speed, inclusive="both")
            ].index,
            axis=0,
        ).copy()
        self.data.df = filtered
        return self.data
