import pytest
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon
from datetime import datetime

from mobiml.datasets import Dataset, SPEED, DIRECTION
from mobiml.transforms.area_aggregator import AreaAggregator


class TestAreaAggregator:
    def setup_method(self):
        # Two points inside polygon A (speeds 2 and 4, directions 30 and 350)
        # One point inside polygon B (speed 6, direction 270)
        # One point outside both polygons (should be ignored)
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(1, 1),
                    "timestamp": datetime(2018, 1, 1, 12, 0, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                    "speed": 2.0,
                    "direction": 30.0,
                },
                {
                    "geometry": Point(1.5, 1.5),
                    "timestamp": datetime(2018, 1, 1, 12, 6, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                    "speed": 4.0,
                    "direction": 350.0,
                },
                {
                    "geometry": Point(6, 6),
                    "timestamp": datetime(2018, 1, 1, 12, 10, 0),
                    "traj_id": 2,
                    "mover_id": 2,
                    "speed": 6.0,
                    "direction": 270.0,
                },
                {
                    "geometry": Point(20, 20),  # outside both polygons
                    "timestamp": datetime(2018, 1, 1, 12, 15, 0),
                    "traj_id": 2,
                    "mover_id": 2,
                    "speed": 8.0,
                    "direction": 45.0,
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=4326)

        # Two equal-sized 3x3 degree polygons
        polygon_a = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        polygon_b = Polygon([(5, 5), (8, 5), (8, 8), (5, 8)])
        polygons_df = pd.DataFrame(
            [
                {"area_name": "A", "geometry": polygon_a},
                {"area_name": "B", "geometry": polygon_b},
            ]
        )
        self.polygons = GeoDataFrame(polygons_df, crs=4326)

    def test_result_is_geodataframe(self):
        dataset = Dataset(self.gdf)
        result = AreaAggregator(dataset).aggregate(self.polygons)

        assert isinstance(result, GeoDataFrame)

    def test_output_has_expected_columns(self):
        dataset = Dataset(self.gdf)
        result = AreaAggregator(dataset).aggregate(self.polygons)

        assert "point_density" in result.columns
        assert "avg_speed" in result.columns
        assert "avg_direction" in result.columns

    def test_output_preserves_polygon_count(self):
        dataset = Dataset(self.gdf)
        result = AreaAggregator(dataset).aggregate(self.polygons)

        assert len(result) == len(self.polygons)

    def test_output_preserves_polygon_attributes(self):
        dataset = Dataset(self.gdf)
        result = AreaAggregator(dataset).aggregate(self.polygons)

        assert "area_name" in result.columns
        assert set(result["area_name"]) == {"A", "B"}

    def test_average_speed_per_area(self):
        dataset = Dataset(self.gdf)
        result = AreaAggregator(dataset).aggregate(self.polygons)
        avg_speed_a = result.loc[result["area_name"] == "A", "avg_speed"].values[0]
        assert avg_speed_a == pytest.approx(3.0)
        avg_speed_b = result.loc[result["area_name"] == "B", "avg_speed"].values[0]
        assert avg_speed_b == pytest.approx(6.0)

    def test_average_direction_per_area(self):
        dataset = Dataset(self.gdf)
        result = AreaAggregator(dataset).aggregate(self.polygons)
        avg_dir_a = result.loc[result["area_name"] == "A", "avg_direction"].values[0]
        assert avg_dir_a == pytest.approx(10.0, abs=0.1)
        avg_dir_b = result.loc[result["area_name"] == "B", "avg_direction"].values[0]
        assert avg_dir_b == pytest.approx(270.0)

    def test_point_density_per_area(self):
        dataset = Dataset(self.gdf)
        result = AreaAggregator(dataset).aggregate(self.polygons)
        density_a = result.loc[result["area_name"] == "A", "point_density"].values[0]
        density_b = result.loc[result["area_name"] == "B", "point_density"].values[0]

        assert density_a == pytest.approx(2 / 9, rel=1e-3)
        assert density_b == pytest.approx(1 / 9, rel=1e-3)
        assert density_a > density_b

    def test_polygon_with_no_points_has_zero_density(self):
        polygon_c = Polygon([(50, 50), (55, 50), (55, 55), (50, 55)])
        polygons_df = pd.DataFrame(
            [
                {
                    "area_name": "A",
                    "geometry": Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
                },
                {"area_name": "C", "geometry": polygon_c},
            ]
        )
        polygons = GeoDataFrame(polygons_df, crs=4326)

        dataset = Dataset(self.gdf)
        result = AreaAggregator(dataset).aggregate(polygons)

        density_c = result.loc[result["area_name"] == "C", "point_density"].values[0]
        assert density_c == pytest.approx(0.0)

    def test_polygon_with_no_points_has_nan_speed_and_direction(self):
        polygon_c = Polygon([(50, 50), (55, 50), (55, 55), (50, 55)])
        polygons_df = pd.DataFrame(
            [
                {
                    "area_name": "A",
                    "geometry": Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
                },
                {"area_name": "C", "geometry": polygon_c},
            ]
        )
        polygons = GeoDataFrame(polygons_df, crs=4326)

        dataset = Dataset(self.gdf)
        result = AreaAggregator(dataset).aggregate(polygons)

        avg_speed_c = result.loc[result["area_name"] == "C", "avg_speed"].values[0]
        avg_dir_c = result.loc[result["area_name"] == "C", "avg_direction"].values[0]
        assert pd.isna(avg_speed_c)
        assert pd.isna(avg_dir_c)

    def test_nan_directions_are_ignored_in_circular_mean(self):
        import math
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(1, 1),
                    "timestamp": datetime(2018, 1, 1, 12, 0, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                    "speed": 2.0,
                    "direction": float("nan"),
                },
                {
                    "geometry": Point(1.5, 1.5),
                    "timestamp": datetime(2018, 1, 1, 12, 6, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                    "speed": 4.0,
                    "direction": 90.0,
                },
            ]
        )
        gdf = GeoDataFrame(df, crs=4326)
        polygon_a = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        polygons = GeoDataFrame(
            [{"area_name": "A", "geometry": polygon_a}], crs=4326
        )

        result = AreaAggregator(Dataset(gdf)).aggregate(polygons)

        avg_dir = result.loc[result["area_name"] == "A", "avg_direction"].values[0]
        assert not math.isnan(avg_dir)
        assert avg_dir == pytest.approx(90.0)


class TestAreaAggregatorTemporal:
    """Tests for spatiotemporal aggregation via the freq parameter."""

    def setup_method(self):
        # Polygon A: one point on day 1, one point on day 2
        # Polygon B: one point on day 1 only
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(1, 1),
                    "timestamp": datetime(2018, 1, 1, 12, 0, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                    "speed": 2.0,
                    "direction": 90.0,
                },
                {
                    "geometry": Point(1.5, 1.5),
                    "timestamp": datetime(2018, 1, 2, 12, 0, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                    "speed": 4.0,
                    "direction": 45.0,
                },
                {
                    "geometry": Point(6, 6),
                    "timestamp": datetime(2018, 1, 1, 12, 0, 0),
                    "traj_id": 2,
                    "mover_id": 2,
                    "speed": 6.0,
                    "direction": 270.0,
                },
            ]
        )
        self.gdf = GeoDataFrame(df, crs=4326)

        polygon_a = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        polygon_b = Polygon([(5, 5), (8, 5), (8, 8), (5, 8)])
        self.polygons = GeoDataFrame(
            [
                {"area_name": "A", "geometry": polygon_a},
                {"area_name": "B", "geometry": polygon_b},
            ],
            crs=4326,
        )

    def test_temporal_result_is_geodataframe(self):
        result = AreaAggregator(Dataset(self.gdf)).aggregate(self.polygons, freq="1D")

        assert isinstance(result, GeoDataFrame)

    def test_temporal_output_has_expected_columns(self):
        result = AreaAggregator(Dataset(self.gdf)).aggregate(self.polygons, freq="1D")

        assert "t" in result.columns
        assert "point_count" in result.columns
        assert "point_density" in result.columns
        assert "avg_speed" in result.columns
        assert "avg_direction" in result.columns

    def test_temporal_result_has_one_row_per_polygon_time_bin(self):
        result = AreaAggregator(Dataset(self.gdf)).aggregate(self.polygons, freq="1D")
        assert len(result) == 3

    def test_temporal_result_preserves_polygon_attributes(self):
        result = AreaAggregator(Dataset(self.gdf)).aggregate(self.polygons, freq="1D")

        assert "area_name" in result.columns

    def test_temporal_point_count_per_area_and_time(self):
        result = AreaAggregator(Dataset(self.gdf)).aggregate(self.polygons, freq="1D")

        day1 = datetime(2018, 1, 1)
        day2 = datetime(2018, 1, 2)

        a_day1 = result.loc[
            (result["area_name"] == "A") & (result["t"] == day1), "point_count"
        ].values[0]
        a_day2 = result.loc[
            (result["area_name"] == "A") & (result["t"] == day2), "point_count"
        ].values[0]
        b_day1 = result.loc[
            (result["area_name"] == "B") & (result["t"] == day1), "point_count"
        ].values[0]

        assert a_day1 == 1
        assert a_day2 == 1
        assert b_day1 == 1

    def test_temporal_average_speed_per_area_and_time(self):
        result = AreaAggregator(Dataset(self.gdf)).aggregate(self.polygons, freq="1D")

        day1 = datetime(2018, 1, 1)
        day2 = datetime(2018, 1, 2)

        a_day1 = result.loc[
            (result["area_name"] == "A") & (result["t"] == day1), "avg_speed"
        ].values[0]
        a_day2 = result.loc[
            (result["area_name"] == "A") & (result["t"] == day2), "avg_speed"
        ].values[0]
        b_day1 = result.loc[
            (result["area_name"] == "B") & (result["t"] == day1), "avg_speed"
        ].values[0]

        assert a_day1 == pytest.approx(2.0)
        assert a_day2 == pytest.approx(4.0)
        assert b_day1 == pytest.approx(6.0)

    def test_temporal_average_direction_per_area_and_time(self):
        result = AreaAggregator(Dataset(self.gdf)).aggregate(self.polygons, freq="1D")

        day1 = datetime(2018, 1, 1)
        day2 = datetime(2018, 1, 2)

        a_day1 = result.loc[
            (result["area_name"] == "A") & (result["t"] == day1), "avg_direction"
        ].values[0]
        a_day2 = result.loc[
            (result["area_name"] == "A") & (result["t"] == day2), "avg_direction"
        ].values[0]

        assert a_day1 == pytest.approx(90.0)
        assert a_day2 == pytest.approx(45.0)

    def test_temporal_point_density_per_area_and_time(self):
        result = AreaAggregator(Dataset(self.gdf)).aggregate(self.polygons, freq="1D")

        day1 = datetime(2018, 1, 1)
        day2 = datetime(2018, 1, 2)

        a_day1 = result.loc[
            (result["area_name"] == "A") & (result["t"] == day1), "point_density"
        ].values[0]
        a_day2 = result.loc[
            (result["area_name"] == "A") & (result["t"] == day2), "point_density"
        ].values[0]

        assert a_day1 == pytest.approx(1 / 9, rel=1e-3)
        assert a_day2 == pytest.approx(1 / 9, rel=1e-3)

    def test_temporal_empty_polygon_time_bins_are_absent(self):
        result = AreaAggregator(Dataset(self.gdf)).aggregate(self.polygons, freq="1D")

        day2 = datetime(2018, 1, 2)

        b_day2 = result.loc[
            (result["area_name"] == "B") & (result["t"] == day2)
        ]
        assert len(b_day2) == 0


class TestAreaAggregatorNoCRS:
    """Tests that AreaAggregator works when dataset and/or polygons have no CRS."""

    def setup_method(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(1, 1),
                    "timestamp": datetime(2018, 1, 1, 12, 0, 0),
                    "traj_id": 1,
                    "mover_id": 1,
                    "speed": 2.0,
                    "direction": 30.0,
                },
                {
                    "geometry": Point(6, 6),
                    "timestamp": datetime(2018, 1, 1, 12, 10, 0),
                    "traj_id": 2,
                    "mover_id": 2,
                    "speed": 6.0,
                    "direction": 270.0,
                },
            ]
        )
        self.gdf_no_crs = GeoDataFrame(df)  # no CRS

        polygon_a = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        polygon_b = Polygon([(5, 5), (8, 5), (8, 8), (5, 8)])
        self.polygons_no_crs = GeoDataFrame(
            [
                {"area_name": "A", "geometry": polygon_a},
                {"area_name": "B", "geometry": polygon_b},
            ]
        )  

    def test_aggregate_without_crs(self):
        result = AreaAggregator(Dataset(self.gdf_no_crs)).aggregate(self.polygons_no_crs)

        assert isinstance(result, GeoDataFrame)
        assert len(result) == 2
        assert result.loc[result["area_name"] == "A", "avg_speed"].values[0] == pytest.approx(2.0)
        assert result.loc[result["area_name"] == "B", "avg_speed"].values[0] == pytest.approx(6.0)
