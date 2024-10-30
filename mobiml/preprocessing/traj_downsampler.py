import pandas as pd
import numpy as np
from mobiml.datasets.utils import TIMESTAMP, TRAJ_ID
from mobiml.datasets._dataset import Dataset
from tqdm.auto import tqdm


class TrajectoryDownsampler:
    def __init__(self, data: Dataset) -> None:
        self.data = data

    def subsample(self, min_dt_sec=10) -> Dataset:
        tqdm.pandas()
        df_clean = (
            self.data.df.sort_values(TIMESTAMP, kind="mergesort")
            .groupby(TRAJ_ID)
            .progress_apply(lambda l: self._subsample_trajectory(l.copy(), min_dt_sec))
            .reset_index(level=0, drop=True)
        )
        self.data.df = df_clean
        return self.data

    def _subsample_trajectory(
        self, traj_df: pd.DataFrame, min_dt_sec=10
    ) -> pd.DataFrame:
        # Based on: https://stackoverflow.com/a/56904899
        sumlm = np.frompyfunc(lambda a, b: a + b if a < min_dt_sec else b, 2, 1)

        traj_dt = traj_df[TIMESTAMP].diff().dt.total_seconds()

        import warnings

        with warnings.catch_warnings(record=True):
            traj_dt_sumlm = sumlm.accumulate(traj_dt, dtype=int)

        return traj_df.drop(traj_dt_sumlm.loc[traj_dt_sumlm < min_dt_sec].index)
