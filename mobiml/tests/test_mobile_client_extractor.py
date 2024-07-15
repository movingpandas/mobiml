import os
import pandas as pd
import geopandas as gpd
from datetime import datetime
from shapely.geometry import Point

from mobiml.datasets import AISDK
from mobiml.transforms import MobileClientExtractor


class TestMobileClientExtractor:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def setup_method(self):
        data = pd.DataFrame(
            [
                {
                    "geometry": Point(1, 2),
                    "# Timestamp": datetime(2018, 1, 1, 12, 0, 0),
                    "MMSI": 1,
                    "SOG": 3,
                },
                {
                    "geometry": Point(2, 2),
                    "# Timestamp": datetime(2018, 1, 1, 12, 2, 0),
                    "MMSI": 1,
                    "SOG": 2,
                },
                {
                    "geometry": Point(3, 0),
                    "# Timestamp": datetime(2018, 1, 1, 12, 4, 0),
                    "MMSI": 1,
                    "SOG": 4,
                },
                {
                    "geometry": Point(4, 4),
                    "# Timestamp": datetime(2018, 1, 1, 12, 7, 0),
                    "MMSI": 1,
                    "SOG": 6,
                },
                {
                    "geometry": Point(6, 5),
                    "# Timestamp": datetime(2018, 1, 1, 12, 14, 0),
                    "MMSI": 1,
                    "SOG": 5,
                },
                {
                    "geometry": Point(9, 9),
                    "# Timestamp": datetime(2018, 1, 1, 12, 16, 0),
                    "MMSI": 1,
                    "SOG": 1,
                },
            ]
        )
        self.aisdk = AISDK(gpd.GeoDataFrame(data, crs=4326))

    def test_two_matching_points(self):
        clients = pd.DataFrame(
            [
                {
                    "geometry": Point(2, 2),
                    "# Timestamp": datetime(2018, 1, 1, 12, 2, 0),
                    "MMSI": "99",
                    "SOG": 1,
                },
                {
                    "geometry": Point(4, 4.000001),
                    "# Timestamp": datetime(2018, 1, 1, 12, 7, 1),
                    "MMSI": "99",
                    "SOG": 1,
                },
            ]
        )
        self.clients = AISDK(gpd.GeoDataFrame(clients, crs=4326))

        antenna_radius_meters = 6000
        expected_pt_count = 2
        extractor = MobileClientExtractor(
            self.aisdk, self.clients, antenna_radius_meters
        )
        assert len(extractor.gdf) == expected_pt_count

    def test_one_matching_point(self):
        clients = pd.DataFrame(
            [
                {
                    "geometry": Point(2, 2),
                    "# Timestamp": datetime(2018, 1, 1, 0, 0, 0),
                    "MMSI": "99",
                    "SOG": 1,
                },
                {
                    "geometry": Point(2, 2.000001),
                    "# Timestamp": datetime(2018, 1, 2, 0, 0, 0),
                    "MMSI": "99",
                    "SOG": 1,
                },
            ]
        )
        self.clients = AISDK(gpd.GeoDataFrame(clients, crs=4326))

        antenna_radius_meters = 3
        expected_pt_count = 1
        extractor = MobileClientExtractor(
            self.aisdk, self.clients, antenna_radius_meters
        )
        assert len(extractor.gdf) == expected_pt_count

    def test_no_match(self):
        clients = pd.DataFrame(
            [
                {
                    "geometry": Point(1, 1),
                    "# Timestamp": datetime(2018, 1, 1, 0, 0, 0),
                    "MMSI": "99",
                    "SOG": 1,
                },
                {
                    "geometry": Point(2, 1),
                    "# Timestamp": datetime(2018, 1, 1, 1, 0, 0),
                    "MMSI": "99",
                    "SOG": 1,
                },
            ]
        )
        self.clients = AISDK(gpd.GeoDataFrame(clients, crs=4326))

        antenna_radius_meters = 3
        expected_pt_count = 0
        extractor = MobileClientExtractor(
            self.aisdk, self.clients, antenna_radius_meters
        )
        assert len(extractor.gdf) == expected_pt_count

    def test_no_spatial_match(self):
        clients = pd.DataFrame(
            [
                {
                    "geometry": Point(1, 1),
                    "# Timestamp": datetime(2018, 1, 1, 0, 0, 0),
                    "MMSI": "99",
                    "SOG": 1,
                },
                {
                    "geometry": Point(2, 1),
                    "# Timestamp": datetime(2018, 1, 2, 0, 0, 0),
                    "MMSI": "99",
                    "SOG": 1,
                },
            ]
        )
        self.clients = AISDK(gpd.GeoDataFrame(clients, crs=4326))

        antenna_radius_meters = 3
        expected_pt_count = 0
        extractor = MobileClientExtractor(
            self.aisdk, self.clients, antenna_radius_meters
        )
        assert len(extractor.gdf) == expected_pt_count

    def test_no_temporal_match(self):
        clients = pd.DataFrame(
            [
                {
                    "geometry": Point(2, 2),
                    "# Timestamp": datetime(2018, 1, 1, 0, 0, 0),
                    "MMSI": "99",
                    "SOG": 1,
                },
                {
                    "geometry": Point(2, 2.000001),
                    "# Timestamp": datetime(2018, 1, 1, 1, 0, 0),
                    "MMSI": "99",
                    "SOG": 1,
                },
            ]
        )
        self.clients = AISDK(gpd.GeoDataFrame(clients, crs=4326))

        antenna_radius_meters = 3
        expected_pt_count = 0
        extractor = MobileClientExtractor(
            self.aisdk, self.clients, antenna_radius_meters
        )
        assert len(extractor.gdf) == expected_pt_count
