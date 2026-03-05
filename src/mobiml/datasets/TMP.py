import pandas as pd
import numpy as np
from datetime import datetime
from mobiml.datasets import Dataset

class SyntheticAIS(Dataset):
    """
    Dataset pour: synthetic_vessel_tracks_with_anomalies_20251007.csv

    Mapping:
      t              -> TIMESTAMP
      AgentID        -> TRAJ_ID & MOVER_ID (copiés par la base à partir des props traj_id/mover_id)
      speed          -> SPEED
      x, y           -> x, y (Lon/Lat en EPSG:4326)
      is_anomaly     -> is_anomaly (conservé)
      anomaly_type   -> anomaly_type (conservé)
      sog      -> calculé (optionnel) si non présent dans le CSV
    """
    name = "Synthetic Vessel Tracks (with anomalies)"
    file_name = "synthetic_vessel_tracks_with_anomalies_20251007.csv"
    source_url = None
    traj_id = "AgentID"
    mover_id = "AgentID"
    crs = 4326

    # Colonnes attendues dans le CSV
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

        # Charge self.df via le mécanisme du Dataset (appelle load_df_from_csv/load_df_from_file)
        super().__init__(path, *args, **kwargs)

        # Filtre bbox si fournie
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

        # Filtre vitesses nulles/négatives (optionnel, comme dans AISDK)
        if "speed" in self.df.columns:
            self.df = self.df[self.df["speed"] > 0]

        # Normalisation des noms/typed du schéma mobiml
        self.df.rename(columns={"speed": SPEED}, inplace=True)
        self.df[TIMESTAMP] = pd.to_datetime(self.df["t"], errors="coerce")
        self.df.drop(columns=["t"], inplace=True)

        # Calcul optionnel du cap (DIRECTION) si absent dans le CSV
        if DIRECTION not in self.df.columns and self.compute_direction:
            # Ordonner par trajectoire puis temps
            self.df.sort_values([self.mover_id, TIMESTAMP], inplace=True)

            # Points suivants par agent
            x_next = self.df.groupby(self.mover_id)["x"].shift(-1)
            y_next = self.df.groupby(self.mover_id)["y"].shift(-1)

            # Bearing géodésique approché
            lon1 = np.radians(self.df["x"].to_numpy())
            lat1 = np.radians(self.df["y"].to_numpy())
            lon2 = np.radians(x_next.to_numpy())
            lat2 = np.radians(y_next.to_numpy())

            # Masque lignes sans "prochain point"
            mask = ~np.isnan(lon2) & ~np.isnan(lat2)

            direction = np.full(len(self.df), np.nan, dtype=float)
            yv = np.sin(lon2[mask] - lon1[mask]) * np.cos(lat2[mask])
            xv = (
                np.cos(lat1[mask]) * np.cos(lat2[mask]) * np.cos(lon2[mask] - lon1[mask])
                + np.sin(lat1[mask]) * np.sin(lat2[mask])
            )
            theta = np.degrees(np.arctan2(yv, xv))
            direction[mask] = (theta + 360.0) % 360.0
            self.df[DIRECTION] = direction

            # Pour la dernière ligne de chaque trajectoire, on remplit à partir de la précédente
            self.df[DIRECTION] = (
                self.df.groupby(self.mover_id)[DIRECTION].ffill().bfill()
            )

        # Colonnes à conserver (on garde aussi is_anomaly/anomaly_type)
        cols_to_keep = [
            traj_id,
            MOVER_ID,
            TIMESTAMP,
            SPEED,
            sog,
            "x",
            "y",
            "is_anomaly",
            "anomaly_type",
            "geometry",   # si la base l’ajoute
            self.mover_id # s'assurer que la colonne source (AgentID) reste disponible
        ]
        existing = [c for c in cols_to_keep if c in self.df.columns]
        for col in list(self.df.columns):
            if col not in existing:
                del self.df[col]

        print(f"{datetime.now()} Loaded DataFrame with {len(self.df)} rows.")

    # Chargeur spécifique CSV — le Dataset devrait l’appeler selon l’extension
    def load_df_from_csv(self, path) -> pd.DataFrame:
        df = pd.read_csv(path, usecols=self.COLS, dtype={"AgentID": str})
        # Filtre bbox au plus tôt pour limiter la mémoire
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

    # Alias de secours si la base appelle un nom générique
    def load_df_from_file(self, path) -> pd.DataFrame:
        return self.load_df_from_csv(path)
