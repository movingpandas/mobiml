from datetime import datetime
import geopandas as gpd
import pandas as pd


class StationaryClientExtractor():
    def __init__(self, dataset, clients_gdf) -> None:
        print(f"{datetime.now()} Converting to GeoDataFrame ...")
        gdf = dataset.to_gdf()
        print(f"{datetime.now()} Computing overlay ...")
        gdf = gdf.overlay(clients_gdf)
        self.gdf = gdf

    def to_feather(self, out_path) -> None:
        self.gdf.to_feather(out_path)