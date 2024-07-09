import os
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from datetime import datetime
from shapely.geometry import Point

from mobiml.datasets.aisdk import AISDK

from mobiml.transforms.stationary_client_extractor import StationaryClientExtractor


class TestStationaryClientExtractor:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def setup_method(self):
        data = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 0),
                    "# Timestamp": datetime(2018, 1, 1, 12, 0, 0),
                    "Longitude": 0,
                    "Latitude": 0,
                    "MMSI": 1,
                    "SOG": 3,
                },
                {
                    "geometry": Point(0, 3),
                    "# Timestamp": datetime(2018, 1, 1, 12, 1, 0),
                    "Longitude": 0,
                    "Latitude": 3,
                    "MMSI": 1,
                    "SOG": 2,
                },
                {
                    "geometry": Point(3, 0),
                    "# Timestamp": datetime(2018, 1, 1, 12, 4, 0),
                    "Longitude": 3,
                    "Latitude": 0,
                    "MMSI": 1,
                    "SOG": 4,
                },
                {
                    "geometry": Point(3, 4),
                    "# Timestamp": datetime(2018, 1, 1, 12, 7, 0),
                    "Longitude": 3,
                    "Latitude": 4,
                    "MMSI": 1,
                    "SOG": 6,
                },
                {
                    "geometry": Point(6, 5),
                    "# Timestamp": datetime(2018, 1, 1, 12, 14, 0),
                    "Longitude": 6,
                    "Latitude": 5,
                    "MMSI": 1,
                    "SOG": 5,
                },
                {
                    "geometry": Point(6, 6),
                    "# Timestamp": datetime(2018, 1, 1, 12, 16, 0),
                    "Longitude": 6,
                    "Latitude": 6,
                    "MMSI": 1,
                    "SOG": 1,
                },
            ]
        )
        self.dataset = GeoDataFrame(data, crs=3857)  # epsg:4326 or epsg:3857?

    def test_stationary_clients(self):
        antennas = ["Point (3 3)"]
        antenna_radius_meters = 4
        df = []
        g = gpd.GeoSeries.from_wkt(antennas)
        buffered_antennas = gpd.GeoDataFrame(
            df, geometry=g, crs=3857
        )  # epsg:4326 or epsg:3857?
        buffered_antennas["geometry"] = buffered_antennas.buffer(antenna_radius_meters)

        min_lon, min_lat, max_lon, max_lat = buffered_antennas.geometry.total_bounds
        sample_aisdk = AISDK(self.dataset, min_lon, min_lat, max_lon, max_lat)

        clients_gdf = StationaryClientExtractor(sample_aisdk, buffered_antennas)
        assert isinstance(clients_gdf, StationaryClientExtractor)

        out_path = os.path.join(
            self.test_dir, "data/test_aisdk_20180208_sample_result.feather"
        )
        clients_gdf.to_feather(out_path)

        test_points = gpd.read_feather(out_path)
        assert len(test_points) == 4

        geometry = [Point(0, 3), Point(3, 0), Point(3, 4), Point(6, 5)]
        assert geometry == test_points["geometry"].tolist()
