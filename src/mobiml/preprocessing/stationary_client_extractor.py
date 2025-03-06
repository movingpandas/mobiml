from datetime import datetime

from mobiml.datasets import Dataset


class StationaryClientExtractor:
    def __init__(self, data: Dataset) -> None:
        self.data = data

    def extract(self, clients_gdf) -> Dataset:
        print(f"{datetime.now()} Converting to GeoDataFrame ...")
        gdf = self.data.to_gdf()
        print(f"{datetime.now()} Computing overlay ...")
        gdf = gdf.overlay(clients_gdf)
        dataset = self.data.copy()
        dataset.df = gdf
        return dataset

    def to_feather(self, out_path) -> None:
        self.data.to_feather(out_path)
