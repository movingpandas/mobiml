import geopandas as gpd
import pandas as pd

from mobiml.datasets import Dataset, SPEED, DIRECTION, TIMESTAMP
from mobiml.utils import circular_mean_degrees


class AreaAggregator:
    def __init__(self, data: Dataset) -> None:
        self.data = data

    def aggregate(
        self, polygons: gpd.GeoDataFrame, freq: str = None
    ) -> gpd.GeoDataFrame:
        """
        Aggregate movement point statistics per polygon area.

        For each polygon (and optionally each time bin), computes the density
        of points per unit area, the average movement speed, and the average
        movement direction.

        Parameters
        ----------
        polygons : GeoDataFrame
            GeoDataFrame of polygon geometries defining the areas to aggregate
            over. Any existing attributes are preserved in the output.
        freq : str, optional
            Pandas-compatible resampling frequency string (e.g. ``"1D"``,
            ``"W"``, ``"ME"``, ``"QE"``, ``"YE"``). When provided, aggregation
            is spatiotemporal: the output contains one row per
            (polygon, time bin) combination that has at least one point.
            When ``None`` (default), aggregation is purely spatial with one
            row per polygon.

        Returns
        -------
        GeoDataFrame with all original polygon columns plus:
        - ``point_count``: number of points within the polygon (and time bin).
          Zero for empty polygons in the spatial-only case; absent from the
          result for empty polygon/time-bin combinations in the temporal case.
        - ``point_density``: ``point_count`` divided by polygon area (in CRS
          units). Note: if the CRS is geographic (e.g. EPSG:4326), area is in
          square degrees, which is not physically meaningful. Reproject
          ``polygons`` to a metric CRS for accurate values.
        - ``avg_speed``: arithmetic mean of point speeds. NaN for empty
          polygons.
        - ``avg_direction``: circular mean of point directions (degrees). NaN
          for polygons where all direction values are missing.
        - ``t`` *(temporal mode only)*: start timestamp of the time bin.

        Examples
        --------
        >>> result = AreaAggregator(dataset).aggregate(polygons_gdf)
        >>> result = AreaAggregator(dataset).aggregate(polygons_gdf, freq="1D")
        """
        gdf = self.data.to_gdf()

        if gdf.crs and polygons.crs and gdf.crs != polygons.crs:
            gdf = gdf.to_crs(polygons.crs)

        joined = gpd.sjoin(gdf, polygons[["geometry"]], how="inner", predicate="within")

        agg_spec = dict(
            point_count=("geometry", "count"),
            avg_speed=(SPEED, "mean"),
            avg_direction=(DIRECTION, circular_mean_degrees),
        )

        if freq is None:
            return self._aggregate_spatially(polygons, joined, agg_spec)
        else:
            return self._aggregate_spatiotemporally(polygons, freq, joined, agg_spec)

    def _aggregate_spatiotemporally(self, polygons, freq, joined, agg_spec):
        stats = (
            joined.groupby(["index_right", pd.Grouper(key=TIMESTAMP, freq=freq)])
            .agg(**agg_spec)
            .reset_index()
        )
        polygon_lookup = polygons.copy()
        polygon_lookup["index_right"] = polygon_lookup.index
        result = gpd.GeoDataFrame(
            stats.merge(polygon_lookup, on="index_right", how="left"),
            crs=polygons.crs,
        )
        result["point_density"] = result["point_count"] / result.geometry.area
        result = result.rename(columns={TIMESTAMP: "t"})
        result = result.drop(columns=["index_right"])
        return result

    def _aggregate_spatially(self, polygons, joined, agg_spec):
        stats = joined.groupby("index_right").agg(**agg_spec)
        result = polygons.copy()
        result["point_count"] = result.index.map(stats["point_count"]).fillna(0)
        result["avg_speed"] = result.index.map(stats["avg_speed"])
        result["avg_direction"] = result.index.map(stats["avg_direction"])
        result["point_density"] = result["point_count"] / result.geometry.area
        return result
