import os
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from datetime import datetime
from shapely.geometry import Point

from mobiml.utils import convert_wgs_to_utm

from mobiml.datasets import AISDK

from mobiml.transforms import StationaryClientExtractor


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
        self.dataset = GeoDataFrame(data, crs=4326)

    def test_stationary_clients(self):
        antennas = ["Point (3 3)"]
        epsg_code = convert_wgs_to_utm(3, 3)
        antenna_radius_meters = 400000
        df = []
        g = gpd.GeoSeries.from_wkt(antennas)
        antennas_gdf = gpd.GeoDataFrame(df, geometry=g, crs=4326)
        antennas_gdf = antennas_gdf.to_crs(epsg_code)
        buffered_antennas = antennas_gdf
        buffered_antennas["geometry"] = antennas_gdf.buffer(antenna_radius_meters)
        buffered_antennas = buffered_antennas.to_crs(4326)

        min_lon, min_lat, max_lon, max_lat = buffered_antennas.geometry.total_bounds
        aisdk = AISDK(self.dataset, min_lon, min_lat, max_lon, max_lat)

        clients_gdf = StationaryClientExtractor(aisdk, buffered_antennas)
        assert isinstance(clients_gdf, StationaryClientExtractor)
        assert len(clients_gdf.gdf) == 4

        geometry = [Point(0, 3), Point(3, 0), Point(3, 4), Point(6, 5)]
        assert geometry == clients_gdf.gdf["geometry"].tolist()

    def test_dataset_not_in_range(self):
        antennas = ["Point (3 3)"]
        epsg_code = convert_wgs_to_utm(3, 3)
        antenna_radius_meters = 100000
        df = []
        g = gpd.GeoSeries.from_wkt(antennas)
        antennas_gdf = gpd.GeoDataFrame(df, geometry=g, crs=4326)
        antennas_gdf = antennas_gdf.to_crs(epsg_code)
        buffered_antennas = antennas_gdf
        buffered_antennas["geometry"] = antennas_gdf.buffer(antenna_radius_meters)
        buffered_antennas = buffered_antennas.to_crs(4326)

        min_lon, min_lat, max_lon, max_lat = buffered_antennas.geometry.total_bounds
        aisdk = AISDK(self.dataset, min_lon, min_lat, max_lon, max_lat)

        clients_gdf = StationaryClientExtractor(aisdk, buffered_antennas)
        assert isinstance(clients_gdf, StationaryClientExtractor)
        assert len(clients_gdf.gdf) == 0
