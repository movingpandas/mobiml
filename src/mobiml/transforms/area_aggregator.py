import geopandas as gpd
from scipy.stats import circmean

from mobiml.datasets import Dataset, SPEED, DIRECTION


class AreaAggregator:
    def __init__(self, data: Dataset) -> None:
        self.data = data

    def aggregate(self, polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Aggregate movement point statistics per polygon area.

        For each polygon, computes the density of points per unit area, the
        average movement speed, and the average movement direction (using
        circular statistics to correctly handle the 0°/360° boundary).

        Parameters
        ----------
        polygons : GeoDataFrame
            GeoDataFrame of polygon geometries defining the areas to aggregate
            over. Any existing attributes are preserved in the output.

        Returns
        -------
        GeoDataFrame with all original polygon columns plus:
        - ``point_count``: total number of points within each polygon. Zero for
          polygons containing no points.
        - ``point_density``: number of points divided by polygon area (in CRS
          units). Zero for polygons containing no points. Note: if the input
          CRS is geographic (e.g. EPSG:4326), area is computed in degrees²,
          which is not physically meaningful. For accurate density values,
          reproject ``polygons`` to a metric CRS before calling this method.
        - ``avg_speed``: arithmetic mean of point speeds within each polygon.
          NaN for polygons containing no points.
        - ``avg_direction``: circular mean of point directions (degrees) within
          each polygon. NaN for polygons containing no points.

        Examples
        --------
        >>> dataset = Dataset(gdf)
        >>> result = AreaAggregator(dataset).aggregate(polygons_gdf)
        """
        gdf = self.data.to_gdf()

        if gdf.crs != polygons.crs:
            gdf = gdf.to_crs(polygons.crs)

        joined = gpd.sjoin(gdf, polygons[["geometry"]], how="inner", predicate="within")

        def _circular_mean_degrees(directions):
            return float(circmean(directions, high=360, low=0))

        stats = joined.groupby("index_right").agg(
            point_count=("geometry", "count"),
            avg_speed=(SPEED, "mean"),
            avg_direction=(DIRECTION, _circular_mean_degrees),
        )

        result = polygons.copy()
        result["point_count"] = result.index.map(stats["point_count"]).fillna(0)
        result["avg_speed"] = result.index.map(stats["avg_speed"])
        result["avg_direction"] = result.index.map(stats["avg_direction"])
        result["point_density"] = result["point_count"] / result.geometry.area

        return result
