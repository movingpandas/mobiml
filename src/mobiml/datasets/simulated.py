import pandas as pd
import numpy as np
from datetime import datetime
from mobiml.datasets import Dataset

class SyntheticAIS(Dataset):
    name = "Synthetic Vessel Tracks (with anomalies)"
    file_name = "synthetic_vessel_2197.csv"
    source_url = None

    # on garde les noms du CSV :
    traj_id = "AgentID"
    mover_id = "AgentID"
    crs = 4326

    COLS = ["t", "x", "y", "AgentID", "speed", "is_anomaly", "anomaly_type"]

    def __init__(
        self,
        path,
        min_lon=None,
        min_lat=None,
        max_lon=None,
        max_lat=None,
        compute_direction: bool = True,
        *args,
        **kwargs,
    ) -> None:
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat
        self.compute_direction = compute_direction

        # Charge self.df via Dataset (qui appelle load_df_from_csv / load_df_from_file)
        super().__init__(path, *args, **kwargs)



        if (
            self.min_lat is not None
            and self.max_lat is not None
            and self.min_lon is not None
            and self.max_lon is not None
        ):
            self.df = self.df[
                (self.df.y >= self.min_lat)
                & (self.df.y <= self.max_lat)
                & (self.df.x >= self.min_lon)
                & (self.df.x <= self.max_lon)
            ]

        # Filtre vitesses nulles/négatives
        if "speed" in self.df.columns:
            self.df = self.df[self.df["speed"] > 0]

        # On garde les noms du CSV, on ajoute juste un timestamp parsé
        if "t" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["t"], errors="coerce")

        # Calcul optionnel de la direction si demandé
        if compute_direction and "x" in self.df.columns and "y" in self.df.columns:
            if "direction" not in self.df.columns:
                # tri par trajectoire + temps
                sort_cols = []
                if self.mover_id in self.df.columns:
                    sort_cols.append(self.mover_id)
                if "timestamp" in self.df.columns:
                    sort_cols.append("timestamp")
                elif "t" in self.df.columns:
                    sort_cols.append("t")

                if sort_cols:
                    self.df.sort_values(sort_cols, inplace=True)

                if self.mover_id in self.df.columns:
                    x_next = self.df.groupby(self.mover_id)["x"].shift(-1)
                    y_next = self.df.groupby(self.mover_id)["y"].shift(-1)
                else:
                    x_next = self.df["x"].shift(-1)
                    y_next = self.df["y"].shift(-1)

                lon1 = np.radians(self.df["x"].to_numpy())
                lat1 = np.radians(self.df["y"].to_numpy())
                lon2 = np.radians(x_next.to_numpy())
                lat2 = np.radians(y_next.to_numpy())

                mask = ~np.isnan(lon2) & ~np.isnan(lat2)

                direction = np.full(len(self.df), np.nan, dtype=float)
                yv = np.sin(lon2[mask] - lon1[mask]) * np.cos(lat2[mask])
                xv = (
                    np.cos(lat1[mask]) * np.cos(lat2[mask]) * np.cos(lon2[mask] - lon1[mask])
                    + np.sin(lat1[mask]) * np.sin(lat2[mask])
                )
                theta = np.degrees(np.arctan2(yv, xv))
                direction[mask] = (theta + 360.0) % 360.0

                self.df["direction"] = direction

                # remplissage des NaN (fin de trajectoire) si on a un id
                if self.mover_id in self.df.columns:
                    self.df["direction"] = (
                        self.df.groupby(self.mover_id)["direction"].ffill().bfill()
                    )

        # Colonnes à conserver : toutes celles du CSV + colonnes calculées utiles
        base_cols = [
            "t",
            "x",
            "y",
            "traj_id",
            "speed",
            "is_anomaly",
            "anomaly_type",
        ]
        extra_cols = ["timestamp", "direction", "geometry"]
        cols_to_keep = [c for c in base_cols + extra_cols if c in self.df.columns]
        self.df = self.df[cols_to_keep]

        print(f"{datetime.now()} Loaded DataFrame with {len(self.df)} rows.")

    def load_df_from_csv(self, path) -> pd.DataFrame:
        df = pd.read_csv(path, usecols=self.COLS, dtype={"AgentID": str})

        if (
            self.min_lat is not None
            and self.max_lat is not None
            and self.min_lon is not None
            and self.max_lon is not None
        ):
            df = df[
                (df.y >= self.min_lat)
                & (df.y <= self.max_lat)
                & (df.x >= self.min_lon)
                & (df.x <= self.max_lon)
            ]
        return df

    def load_df_from_file(self, path) -> pd.DataFrame:
        return self.load_df_from_csv(path)
