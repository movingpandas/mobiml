import os
import pandas as pd
import geopandas as gpd
from copy import deepcopy
from geopandas import GeoDataFrame
from datetime import datetime
from shapely.geometry import Point

from mobiml.datasets.aisdk import AISDK

from mobiml.transforms.mobile_client_extractor import MobileClientExtractor


class TestMobileClientExtractor:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def setup_method(self):
        data = pd.DataFrame(
            [
                {
                    "geometry": Point(1, 2),
                    "# Timestamp": datetime(2018, 1, 1, 12, 0, 0),
                    "Longitude": 1,
                    "Latitude": 2,
                    "MMSI": 1,
                    "SOG": 3,
                },
                {
                    "geometry": Point(2, 2),
                    "# Timestamp": datetime(2018, 1, 1, 12, 2, 0),
                    "Longitude": 2,
                    "Latitude": 2,
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
                    "geometry": Point(9, 9),
                    "# Timestamp": datetime(2018, 1, 1, 12, 16, 0),
                    "Longitude": 9,
                    "Latitude": 9,
                    "MMSI": 1,
                    "SOG": 1,
                },
            ]
        )
        self.dataset = GeoDataFrame(data, crs=4326)

        client = pd.DataFrame(
            [
                {
                    "geometry": Point(2, 2),
                    "# Timestamp": datetime(2018, 1, 1, 0, 0, 0),
                    "Longitude": 2,
                    "Latitude": 2,
                    "MMSI": 99,
                    "SOG": 1,
                },
                {
                    "geometry": Point(2, 2.000001),
                    "# Timestamp": datetime(2018, 1, 2, 0, 0, 0),
                    "Longitude": 2,
                    "Latitude": 2.000001,
                    "MMSI": 99,
                    "SOG": 1,
                },
            ]
        )
        self.clients = GeoDataFrame(client, crs=4326)

    def test_mobile_clients(self):
        antenna_radius_meters = 3

        aisdk = AISDK(self.dataset)
        clients = AISDK(self.clients)

        # vessels = deepcopy(
        # aisdk
        # )  # AISDK(path, min_lon, min_lat, max_lon, max_lat, vessel_type)

        client_gdf = MobileClientExtractor(aisdk, clients, antenna_radius_meters)
        print(client_gdf)
        assert False
