from os.path import exists, splitext
from datetime import datetime, timedelta
from shapely import Point
import pandas as pd 
import geopandas as gpd 
import movingpandas as mpd


TRAJ_ID = 'traj_id'
MOVER_ID = 'mover_id'
TIMESTAMP = 'timestamp'
COORDS = 'coordinates'
ROWNUM = 'running_number'
SPEED = 'speed'


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
    except:
        return None 

def get_x_from_xy(df, xycol=COORDS) -> pd.Series:
    return df[xycol].apply(lambda xy: val_or_none(xy, 0))


def get_y_from_xy(df, xycol=COORDS) -> pd.Series:
    return df[xycol].apply(lambda xy: val_or_none(xy, 1))


def get_point_from_xy(df, xycol=COORDS) -> pd.Series:
    return df[xycol].apply(create_point)


def get_point_from_x_y(df, xcol='x', ycol='y') -> pd.Series:
    return df.apply(lambda row: Point(row[xcol], row[ycol]), axis=1)


class Dataset():
    name = None
    file_name = None
    source_url = None
    traj_id = None
    mover_id = None
    speed = None
    crs = None 
    running_number_added = False

    def __init__(self, path, *args, **kwargs) -> None:
        if not exists(path):
            raise ValueError(f"""
            Please specify the path to the {self.name} {self.file_name}
            You may download this dataset from: {self.source_url}
            """)
        
        ext = splitext(self.file_name)[1]
        if ext == ".pickle":
            nrows = kwargs.pop("nrows", None)
            df = pd.read_pickle(path, *args, **kwargs)
            if nrows: 
                df = df.head(nrows)
        elif ext == ".csv":
            df = pd.read_csv(path, *args, **kwargs)
        else:
            df = gpd.read_file(path)
        
        df.rename(columns={self.traj_id: TRAJ_ID}, inplace=True)
        if self.mover_id:
            if self.traj_id == self.mover_id:
                df[MOVER_ID] = df[TRAJ_ID]
            else:
                df.rename(columns={self.mover_id: MOVER_ID}, inplace=True)

        self.df = df

    def merge_xcol_and_ycol_to_xycol(self, xcol, ycol) -> None:
        self.df[COORDS] = self.df.apply(
            lambda row: list(zip(row[xcol], row[ycol])), axis=1)
        self.df.drop(columns=[xcol, ycol], inplace=True)

    def explode_coordinate_list(self, coords=COORDS) -> None:
        self.df = self.df.explode(coords)
        self.df[ROWNUM] = self.df.groupby(TRAJ_ID).cumcount()
        self.running_number_added = True

    def to_df(self) -> pd.DataFrame:
        df = self.df.copy()
        if type(df) == gpd.GeoDataFrame:
            df['x'] = df.geometry.x
            df['y'] = df.geometry.y
        elif not ('x' in df.columns) and not ('y' in df.columns):
            df['x'] = get_x_from_xy(df) 
            df['y'] = get_y_from_xy(df) 
            df.drop(columns=[COORDS], inplace=True)
        if self.running_number_added:
            df.drop(columns=[ROWNUM], inplace=True)
        return df

    def to_gdf(self) -> gpd.GeoDataFrame:
        df = self.df.copy()
        if type(df) == gpd.GeoDataFrame:
            return df 
        if COORDS in df.columns:
            df['geometry'] = get_point_from_xy(df)  
            df.drop(columns=[COORDS], inplace=True)
        else:
            df['geometry'] = get_point_from_x_y(df)
            df.drop(columns=['x', 'y'], inplace=True)
        if self.running_number_added:
            df.drop(columns=[ROWNUM], inplace=True)
        gdf = gpd.GeoDataFrame(df, crs=self.crs)
        return gdf

    def to_trajs(self) -> mpd.TrajectoryCollection:
        gdf = self.to_gdf()
        trajs = mpd.TrajectoryCollection(
            gdf, traj_id_col=TRAJ_ID, obj_id_col=MOVER_ID, t=TIMESTAMP, crs=self.crs)
        return trajs

    def plot(self, *args, **kwargs):
        df = self.to_df()
        return df.plot.scatter(x='x', y='y', *args, **kwargs)

    def datashade(self, *args, **kwargs):
        import datashader as ds
        from hvplot import pandas  # required for df.hvplot
        from holoviews import opts
        from holoviews.element import tiles
        opts.defaults(opts.Overlay(active_tools=['wheel_zoom']))
        BG_TILES = tiles.CartoLight()
        df = self.to_df()
        if self.crs is None:
            return df.hvplot.scatter(x='x', y='y', datashade=True, *args, **kwargs)
        if self.crs != 4326:
            # TODO: reproject
            pass
        df.loc[:, 'x'], df.loc[:, 'y'] = ds.utils.lnglat_to_meters(df.x, df.y)
        plot = df.hvplot.scatter(x='x', y='y', datashade=True, *args, **kwargs)
        return BG_TILES * plot 


class MovebankGulls(Dataset):
    name = "Movebank Migrating Gulls"
    file_name = "gulls.gpkg"
    source_url = "https://github.com/movingpandas/movingpandas-examples/blob/main/data/gulls.gpkg"
    traj_id = 'individual-local-identifier'
    mover_id = 'individual-local-identifier'
    crs = 4326

    def __init__(self, path, drop_extra_cols=True, *args, **kwargs) -> None:
        super().__init__(path, *args, **kwargs)
        self.df.rename(columns={'timestamp': TIMESTAMP}) 
        if drop_extra_cols:
            self.df.drop(columns=["individual-taxon-canonical-name", "study-name", 
                "location-long", "location-lat", "event-id", "visible"], inplace=True)
        print(f"Loaded Dataframe with {len(self.df)} rows.")


class BrestAIS(Dataset):
    name = "Brest AIS"
    file_name = "nari_dynamic_sar.csv or nari_dynamic.csv"
    source_url = "https://zenodo.org/record/1167595/files/%5BP1%5D%20AIS%20Data.zip?download=1"
    traj_id = 'sourcemmsi'
    mover_id = 'sourcemmsi'
    crs = 4326

    def __init__(self, path, *args, **kwargs) -> None:
        super().__init__(path, *args, **kwargs)
        self.df.rename(columns={
            'ts': 't', 'lon':'x', 'lat': 'y', 'speedoverground': SPEED}, 
            inplace=True)
        self.df[TIMESTAMP] = pd.to_datetime(self.df['t'], unit='s')
        self.df.drop(columns=['t'], inplace=True)
        print(f"Loaded Dataframe with {len(self.df)} rows.")


class CopenhagenCyclists(Dataset):
    name = "Copenhagen Cyclists"
    file_name = "df_bike.pickle"
    source_url = "https://zenodo.org/record/7288616"
    traj_id = 'id'
    mover_id = None
    crs = None

    def __init__(self, path, drop_extra_cols=True, *args, **kwargs) -> None:
        super().__init__(path, *args, **kwargs)

        def _compute_datetime(row) -> datetime:
            # some educated guessing going on here: 
            # the paper states that the video covers 2021-06-09 07:00-08:00
            t0 = datetime(2021,6,9,7,0,0)
            offset = (row['frame_in'] + row[ROWNUM]) * timedelta(seconds=2)
            return t0 + offset 

        self.merge_xcol_and_ycol_to_xycol('xs_640x360', 'ys_640x360')
        self.explode_coordinate_list()
        self.df[TIMESTAMP] = self.df.apply(_compute_datetime, axis=1)
        self.df.drop(columns=['frame_in'], inplace=True)
        if drop_extra_cols:
            self.df.drop(columns=["frame_out", "num_frames", "time_on_screen_s",
                "x_start_640x360", "x_end_640x360", "y_start_640x360", "y_end_640x360", 
                "class"], inplace=True)
        print(f"Loaded Dataframe with {len(self.df)} rows.")


class PortoTaxis(Dataset):
    name = "Porto Taxi"
    file_name = "train.csv or test.csv"
    source_url = "https://www.kaggle.com/competitions/pkdd-15-predict-taxi-service-trajectory-i/data"
    traj_id = "TRIP_ID"
    mover_id = "TAXI_ID"
    crs = 4326

    def __init__(self, path, *args, **kwargs) -> None:
        super().__init__(path, *args, **kwargs)

        def _compute_datetime(row):
            t0 = unixtime_to_datetime(row['TIMESTAMP'])
            offset = row[ROWNUM] * timedelta(seconds=15)
            return t0 + offset

        self.df.POLYLINE = self.df.POLYLINE.apply(eval)  # string to list
        self.df.rename(columns={'POLYLINE': COORDS}, inplace=True)
        self.explode_coordinate_list()
        self.df[TIMESTAMP] = self.df.apply(_compute_datetime, axis=1)
        self.df.drop(columns=['TIMESTAMP'], inplace=True)
        print(f"Loaded Dataframe with {len(self.df)} rows.")
