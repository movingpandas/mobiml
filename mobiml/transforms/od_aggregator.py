import pandas as pd
import movingpandas as mpd
from datetime import datetime

from mobiml.datasets import _Dataset, MOVER_ID, TIMESTAMP, TRAJ_ID

try:
    import h3
except ImportError as error:
    raise ImportError(
        "Missing optional dependencies. To use the ODAggregator please "
        "install h3-py using conda install conda-forge::h3-py"
    ) from error


class ODAggregator:
    def __init__(self, data: _Dataset) -> None:
        self.data = data

    def get_od_for_h3(self, res, freq) -> pd.DataFrame:
        """
        Extract start and end points (OD) for trajectories from a Dataset and aggregate
        them in H3.
        H3 level and temporal binning are customizable.

        Parameters
        ----------
        res : int
            Desired number for the H3 resolution
        freq : string
            Desired frequency for temporal binning, guide:
            https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases

        Returns
        ----------
        DataFrame with columns t, h3_cell, origin, destination

        Examples
        ----------
        >>> taxis = PortoTaxis(r"../examples/data/train.csv", nrows=10000)
        >>> od_h3 = ODAggregator(taxis).get_od_for_h3(7, "1h")
        """
        gdf = self.data.to_gdf()
        gdf = gdf[[TRAJ_ID, MOVER_ID, TIMESTAMP, "geometry"]]

        gdf["x"] = gdf.geometry.x
        gdf["y"] = gdf.geometry.y

        print(f"{datetime.now()} Identifying h3 cell id ...")

        def get_cell(row):
            # Updated for h3 > 4.0 https://github.com/uber/h3-py/issues/100
            if not row.geometry:
                cell = None
            else:
                cell = h3.latlng_to_cell(row.y, row.x, res)
            return str(cell)

        gdf["h3_cell"] = gdf.apply(get_cell, axis=1)
        gdf = gdf[gdf.h3_cell != "0"]

        tc = mpd.TrajectoryCollection(
            gdf,
            traj_id_col=TRAJ_ID,
            obj_id_col=MOVER_ID,
            t=TIMESTAMP,
            x="x",
            y="y",
        )

        print(f"{datetime.now()} Getting start locations ...")
        start = tc.get_start_locations()
        start = start.rename(
            columns={
                TIMESTAMP: "start_t",
                "geometry": "origin",
                "h3_cell": "h3_cell_origin",
            }
        )
        start = start.drop(columns={MOVER_ID, "x", "y"})

        start = start.set_index("start_t")
        df = start.groupby([pd.Grouper(freq=freq), "h3_cell_origin"]).count()
        df = df.origin
        df = df.reset_index()

        print(f"{datetime.now()} Getting end locations ...")
        end = tc.get_end_locations()
        end = end.rename(
            columns={
                TIMESTAMP: "end_t",
                "geometry": "destination",
                "h3_cell": "h3_cell_destination",
            }
        )
        end = end.drop(columns={TRAJ_ID, MOVER_ID, "x", "y"})

        end = end.set_index("end_t")
        df1 = end.groupby([pd.Grouper(freq=freq), "h3_cell_destination"]).count()
        df1 = df1.destination
        df1 = df1.reset_index()

        combined = pd.concat([df, df1], axis=1)
        combined["t"] = combined[["start_t", "end_t"]].max(axis=1)
        combined["h3_cell"] = combined["h3_cell_origin"]
        combined["h3_cell"] = combined["h3_cell"].fillna(
            combined["h3_cell_destination"]
        )
        combined["origin"] = combined["origin"].fillna(0)
        combined["origin"] = combined["origin"].astype(int)
        combined["destination"] = combined["destination"].fillna(0)
        combined = combined[["t", "h3_cell", "origin", "destination"]]

        return combined
