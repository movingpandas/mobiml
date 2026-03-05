import pytest
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon
from datetime import datetime

from mobiml.datasets import Dataset, SPEED, DIRECTION
from mobiml.transforms.area_aggregator import AreaAggregator


class TestAreaAggregator:
    """Tests for AreaAggregator, which aggregates movement point statistics per polygon area."""

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

        # Polygon A: two points with speeds 2.0 and 4.0 -> mean = 3.0
        avg_speed_a = result.loc[result["area_name"] == "A", "avg_speed"].values[0]
        assert avg_speed_a == pytest.approx(3.0)

        # Polygon B: one point with speed 6.0 -> mean = 6.0
        avg_speed_b = result.loc[result["area_name"] == "B", "avg_speed"].values[0]
        assert avg_speed_b == pytest.approx(6.0)

    def test_average_direction_per_area(self):
        dataset = Dataset(self.gdf)
        result = AreaAggregator(dataset).aggregate(self.polygons)

        # Polygon A: two points with directions 30.0 and 350.0.
        # Arithmetic mean would be 190.0 (wrong — crosses 0°/360°).
        # Circular mean: atan2(sin(30°)+sin(350°), cos(30°)+cos(350°)) ≈ 10.0°
        avg_dir_a = result.loc[result["area_name"] == "A", "avg_direction"].values[0]
        assert avg_dir_a == pytest.approx(10.0, abs=0.1)

        # Polygon B: one point with direction 270.0 -> mean = 270.0
        avg_dir_b = result.loc[result["area_name"] == "B", "avg_direction"].values[0]
        assert avg_dir_b == pytest.approx(270.0)

    def test_point_density_per_area(self):
        dataset = Dataset(self.gdf)
        result = AreaAggregator(dataset).aggregate(self.polygons)

        # Both polygons are 3x3 = 9 sq units (in CRS units).
        # Polygon A has 2 points -> density = 2/9
        # Polygon B has 1 point  -> density = 1/9
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
