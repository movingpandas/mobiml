from pandas import DataFrame
from mobiml.datasets.utils import TIMESTAMP


def trajectorycollection_to_df(trajs):
    gdf = trajs.to_point_gdf()
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    gdf[TIMESTAMP] = gdf.index
    df = DataFrame(gdf.drop(columns="geometry")).reset_index(level=0, drop=True)

    return df
