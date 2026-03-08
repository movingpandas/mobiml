import os
from datetime import datetime

import geopandas as gpd
import pandas as pd
import pymeos_cffi.errors as pymeos_errors
from pytest import raises
from shapely.geometry import Point

from mobiml.datasets import AISDK
from mobiml.preprocessing import MobileClientExtractor
from time import tzname

try:
    from pymeos import pymeos_initialize, TGeogPointInst, TGeogPointSeq
except ImportError as error:
    raise ImportError(
        "Missing optional dependencies. To use the MobileClientExtractor please "
        "install pymeos"
    ) from error

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
        extractor = MobileClientExtractor(self.aisdk)
        client_data = extractor.extract(self.clients, antenna_radius_meters)
        assert isinstance(client_data, AISDK)
        assert len(client_data.df) == expected_pt_count

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
        extractor = MobileClientExtractor(self.aisdk)
        client_data = extractor.extract(self.clients, antenna_radius_meters)
        assert isinstance(client_data, AISDK)
        assert len(client_data.df) == expected_pt_count

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
        extractor = MobileClientExtractor(self.aisdk)
        client_data = extractor.extract(self.clients, antenna_radius_meters)
        assert isinstance(client_data, AISDK)
        assert len(client_data.df) == expected_pt_count

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
        extractor = MobileClientExtractor(self.aisdk)
        client_data = extractor.extract(self.clients, antenna_radius_meters)
        assert isinstance(client_data, AISDK)
        assert len(client_data.df) == expected_pt_count

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
        extractor = MobileClientExtractor(self.aisdk)
        client_data = extractor.extract(self.clients, antenna_radius_meters)
        assert isinstance(client_data, AISDK)
        assert len(client_data.df) == expected_pt_count

    def test_daylight_saving_data(self):
        tz_unaware_client_data = pd.DataFrame(

            [
                {
                    "geometry": Point(2, 2),
                    "# Timestamp": datetime(2018, 3, 25, 2, 5, 0),
                    "MMSI": "99",
                    "SOG": 1,
                },
                {
                    "geometry": Point(2, 2.000001),
                    "# Timestamp": datetime(2018, 3, 25, 3, 0, 0),
                    "MMSI": "99",
                    "SOG": 1,
                },
            ]
        )
        self.clients = AISDK(gpd.GeoDataFrame(tz_unaware_client_data, crs=4326))

        extractor = MobileClientExtractor(self.aisdk)

        if 'UTC' in tzname:
            _ = extractor.extract(self.clients, 3)
        else:
            with raises(pymeos_errors.MeosInvalidArgValueError):
                _ = extractor.extract(self.clients, 3)

        # if tz are set, they are currently dropped by movingpandas during trajectory creation
        self.clients.df.timestamp = self.clients.df.timestamp.dt.tz_localize('UTC')

        if 'UTC' in tzname:
            _ = extractor.extract(self.clients, 3)
        else:
            with raises(pymeos_errors.MeosInvalidArgValueError):
                _ = extractor.extract(self.clients, 3)

    def test_pymeos(self):
        """
        two AIS timestamps 02:05:00, 03:00:00, no tz set, UTC implied

        daylight saving starts on 2018-03-25 02:00

        MeosInvalidArgValueError (12): Timestamps for temporal value must be increasing: 2018-03-25 03:05:00+02, 2018-03-25 03:00:00+02
        """

        wkt_utc = "[POINT (11.64066 57.602362)@2018-03-25 02:05:00+00:00, POINT (11.640432 57.602283)@2018-03-25 03:00:00+00:00]"

        TGeogPointSeq(string=wkt_utc, normalize=False)

        wkt_unaware = "[POINT (11.64066 57.602362)@2018-03-25 02:05:00, POINT (11.640432 57.602283)@2018-03-25 03:00:00]"

        if 'UTC' in tzname:
            TGeogPointSeq(string=wkt_unaware, normalize=False)
        else:
            with raises(pymeos_errors.MeosInvalidArgValueError):
                TGeogPointSeq(string=wkt_unaware, normalize=False)


