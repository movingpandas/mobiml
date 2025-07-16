"""
Based on https://github.com/DataStories-UniPi/Nautilus

As presented in Andreas Tritsarolis, Nikos Pelekis, Konstantina Bereta, Dimitris Zissis,
and Yannis Theodoridis. 2024. On Vessel Location Forecasting and the Effect of Federated
Learning. In Proceedings of the 25th Conference on Mobile Data Management (MDM).
"""

from pandas import DataFrame, merge, Series
from mobiml.datasets import Dataset, SPEED, DIRECTION, TIMESTAMP, TRAJ_ID
from mobiml.utils import applyParallel, shapely_coords_numpy

import warnings
warnings.filterwarnings('ignore')


class DeltaDatasetCreator:
    def __init__(self, data: Dataset) -> None:
        self.data = data
        self.input_feats = ["dx_curr", "dy_curr", "dt_curr", "dt_next"]
        self.output_feats = ["dx_next", "dy_next"]

    def get_delta_dataset(self, col=None, njobs=50) -> DataFrame:
        traj_delta = applyParallel(
            self.data.to_gdf().groupby([TRAJ_ID, col], group_keys=True),
            lambda l: self.create_delta_dataset(l),
            n_jobs=njobs,
        )
        return traj_delta

    def create_delta_dataset(self, segment, crs=3857, min_pts=22) -> DataFrame:
        if len(segment) < min_pts:
            return None

        segment.sort_values(TIMESTAMP, inplace=True)

        delta_curr = self.compute_x_y_deltas(segment, crs)
        delta_curr_feats = self.compute_speed_direction_deltas(segment)
        delta_next = delta_curr.shift(-1)
        delta_tau = merge(
            segment[TIMESTAMP].diff().dt.total_seconds().rename("dt_curr"),
            segment[TIMESTAMP].diff().dt.total_seconds().shift(-1).rename("dt_next"),
            right_index=True,
            left_index=True,
        )

        return (
            delta_curr.join(delta_curr_feats)
            .join(delta_tau)
            .join(delta_next, lsuffix="_curr", rsuffix="_next")
            .dropna(subset=["dt_curr", "dt_next"])
            .fillna(method="bfill")
        )

    def compute_speed_direction_deltas(self, segment):
        delta_curr_feats = (
            segment[[SPEED, DIRECTION]]
            .diff()
            .rename({SPEED: "dspeed_curr", DIRECTION: "dcourse_curr"}, axis=1)
        )

        return delta_curr_feats

    def compute_x_y_deltas(self, segment, crs):
        delta_curr = (
            segment.to_crs(crs)
            .geometry.apply(
                lambda l: Series(shapely_coords_numpy(l), index=["dx", "dy"])
            )
            .diff()
        )

        return delta_curr

    def get_windowed_dataset(self, col=None, njobs=50):
        """
        Get constant-length windows of delta values for ML model training
        """

        traj_delta = self.get_delta_dataset(col=col, njobs=njobs)

        traj_delta_windows = (
            applyParallel(
                traj_delta.reset_index().groupby([TRAJ_ID, col]),
                lambda l: self.traj_windowing(l),
                n_jobs=njobs,
            )
            .reset_index(level=-1)
            .pivot(columns=["level_2"])
            .rename_axis([None, None], axis=1)
            .sort_index(axis=1, ascending=False)
        )

        traj_delta_windows.columns = traj_delta_windows.columns.droplevel(0)
        traj_delta_windows = traj_delta_windows.explode(["samples", "labels"])

        return traj_delta_windows

    def traj_windowing(self, segment, length_max=1024, length_min=20, stride=512):
        traj_inputs, traj_labels = [], []

        output_feats_idx = [
            segment.columns.get_loc(output_feat) for output_feat in self.output_feats
        ]

        for ptr_curr in range(0, len(segment), stride):
            segment_window = segment.iloc[ptr_curr : ptr_curr + length_max].copy()

            if len(segment_window) < length_min:
                break

            traj_inputs.append(segment_window[self.input_feats].values)
            traj_labels.append(segment_window.iloc[-1, output_feats_idx].values)

        return Series([traj_inputs, traj_labels], index=["samples", "labels"])
