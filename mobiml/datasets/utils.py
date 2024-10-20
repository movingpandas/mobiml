from datetime import datetime
from shapely import Point
import pandas as pd

TRAJ_ID = "traj_id"
MOVER_ID = "mover_id"
TIMESTAMP = "timestamp"
COORDS = "coordinates"
ROWNUM = "running_number"
SPEED = "speed"
DIRECTION = "direction"


def unixtime_to_datetime(unix_time) -> datetime:
    return datetime.fromtimestamp(unix_time)


def create_point(xy) -> Point:
    try:
        return Point(xy)
    except TypeError:  # when there are nan values in the input data
        return None


def val_or_none(xy, idx):
    try:
        return xy[idx]
    except IndexError:
        return None


def get_x_from_xy(df, xycol=COORDS) -> pd.Series:
    return df[xycol].apply(lambda xy: val_or_none(xy, 0))


def get_y_from_xy(df, xycol=COORDS) -> pd.Series:
    return df[xycol].apply(lambda xy: val_or_none(xy, 1))


def get_point_from_xy(df, xycol=COORDS) -> pd.Series:
    return df[xycol].apply(create_point)


def get_point_from_x_y(df, xcol="x", ycol="y") -> pd.Series:
    return df.apply(lambda row: Point(row[xcol], row[ycol]), axis=1)
