import h3
import pandas as pd
import movingpandas as mpd
from datetime import datetime

from mobiml.datasets import Dataset


class ODAggregator:
    def __init__(self, data: Dataset) -> None:
        self.data = data

    def get_od_for_h3(self, res) -> Dataset:
        gdf = self.data.to_gdf()
        gdf["x"] = gdf.geometry.x
        gdf["y"] = gdf.geometry.y

        print(f"{datetime.now()} Identifying h3 cell id ...")
        # Based on: https://medium.com/@jesse.b.nestler/how-to-convert-h3-cell-boundaries-to-shapely-polygons-in-python-f7558add2f63
        gdf["h3_cell"] = gdf.apply(
            lambda row: str(h3.geo_to_h3(row.y, row.x, res)), axis=1
        )

        gdf = gdf[gdf.h3_cell != "0"]

        gdf = gdf.drop(
            columns={
                "CALL_TYPE",
                "ORIGIN_CALL",
                "ORIGIN_STAND",
                "DAY_TYPE",
                "MISSING_DATA",
            }
        )

        tc = mpd.TrajectoryCollection(gdf, "traj_id", t="timestamp", x="x", y="y")

        print(f"{datetime.now()} Getting start locations ...")
        start = tc.get_start_locations()
        start = start.rename(
            columns={
                "timestamp": "start_t",
                "geometry": "origin",
                "h3_cell": "h3_cell_origin",
            }
        )
        start = start.drop(columns={"mover_id", "x", "y"})

        print(f"{datetime.now()} Getting end locations ...")
        end = tc.get_end_locations()
        end = end.rename(
            columns={
                "timestamp": "end_t",
                "geometry": "destination",
                "h3_cell": "h3_cell_destination",
            }
        )
        end = end.drop(columns={"traj_id", "mover_id", "x", "y"})

        od_h3 = pd.concat([start, end], axis=1)

        self.data.df = od_h3
        return self.data
