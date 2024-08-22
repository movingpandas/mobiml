from typing import Tuple, List
import numpy as np
import math


# Get the coordinates of a Shapely Geometry (e.g. Point, Polygon, etc.) as NumPy array
shapely_coords_numpy = lambda l: np.array(*list(l.coords))


class XY(Tuple):
    def __init__(x: np.ndarray, y: np.ndarray):
        super.__init__(x, y)


class Dataset(Tuple):
    def __init__(a: XY, b: XY):
        super.__init__(a, b)


class LogRegParams(Tuple):
    def __init__(a: XY, b: Tuple[np.ndarray]):
        super.__init__(a, b)


class XYList(List):
    def __init__(a: XY):
        super.__init__(a)


def applyParallel(df_grouped, fun, n_jobs=-1, **kwargs):
    from pandas import concat
    import tqdm
    import multiprocessing
    from joblib import delayed, Parallel

    """
    Forked from: https://stackoverflow.com/a/27027632
    """
    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    print(f"Scaling {fun} to {n_jobs} CPUs")

    df_grouped_names = df_grouped.grouper.names

    _fun = lambda name, group: (
        fun(group.drop(df_grouped_names, axis=1)),
        name,
    )

    result, keys = zip(
        *Parallel(n_jobs=n_jobs)(
            delayed(_fun)(name, group)
            for name, group in tqdm.tqdm(df_grouped, **kwargs)
        )
    )
    import warnings

    with warnings.catch_warnings(record=True):
        concatenated = concat(result, keys=keys, names=df_grouped_names)
    return concatenated


def convert_wgs_to_utm(lon, lat):
    """Forked from: https://stackoverflow.com/a/40140326"""
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = "0" + utm_band
    if lat >= 0:
        epsg_code = "326" + utm_band
    else:
        epsg_code = "327" + utm_band
    return epsg_code
