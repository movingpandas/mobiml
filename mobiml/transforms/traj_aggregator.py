from datetime import datetime
from itertools import groupby

import pandas as pd

from mobiml.datasets import MOVER_ID, SPEED, DIRECTION
from mobiml.datasets.aisdk import SHIPTYPE

try:
    import h3pandas as h3  # noqa F401
except ImportError as error:
    raise ImportError(
        "Missing optional dependencies. To use the TrajectoryAggregator "
        "please install h3pandas"
    ) from error


class TrajectoryAggregator:
    def __init__(self, trajs, vessels) -> None:
        self.trajs = trajs
        self.vessels = vessels

    def aggregate_trajs(self, h3_resolution) -> pd.DataFrame:
        print(f"{datetime.now()} Enriching trajectories ...")
        traj_gdf = self.trajs.to_traj_gdf(
            agg={"client": "mode", MOVER_ID: "mode", SPEED: ["max", "median"]}
        )
        traj_gdf.rename(
            columns={"client_mode": "client", f"{MOVER_ID}_mode": MOVER_ID},
            inplace=True,
        )
        traj_gdf["H3_seq"] = [
            traj_to_h3_sequence(traj, h3_resolution) for traj in self.trajs.trajectories
        ]

        start_locations = self.trajs.get_start_locations()
        end_locations = self.trajs.get_end_locations()
        traj_gdf[f"{SPEED}_start"] = start_locations[SPEED].values
        traj_gdf[f"{DIRECTION}_start"] = start_locations[DIRECTION].values
        traj_gdf["x_start"] = start_locations.geometry.x.values
        traj_gdf["y_start"] = start_locations.geometry.y.values
        traj_gdf[f"{SPEED}_end"] = end_locations[SPEED].values
        traj_gdf[f"{DIRECTION}_end"] = end_locations[DIRECTION].values
        traj_gdf["x_end"] = end_locations.geometry.x.values
        traj_gdf["y_end"] = end_locations.geometry.y.values

        cols = traj_gdf.columns.to_list()
        cols.append(SHIPTYPE)
        dataset = traj_gdf.merge(
            right=self.vessels, how="left", left_on=MOVER_ID, right_index=True
        )
        dataset = dataset[cols]
        dataset.dropna(inplace=True)

        print(f"Enriched dataset columns: {dataset.columns}")
        return dataset


@staticmethod
def traj_to_h3_sequence(my_traj, h3_resolution):
    df = my_traj.df.copy()
    df["t"] = df.index
    df["h3_cell"] = df.h3.geo_to_h3(resolution=h3_resolution).index
    h3_sequence = [key for key, _ in groupby(df.h3_cell.values.tolist())]
    return h3_sequence
