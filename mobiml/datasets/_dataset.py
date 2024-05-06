from os.path import exists, splitext
from datetime import datetime
from shapely import Point
import pandas as pd 
import geopandas as gpd 
import movingpandas as mpd
from zipfile import ZipFile


TRAJ_ID = 'traj_id'
MOVER_ID = 'mover_id'
TIMESTAMP = 'timestamp'
COORDS = 'coordinates'
ROWNUM = 'running_number'
SPEED = 'speed'
DIRECTION = 'direction'


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
    timestamp = None
    speed = None
    crs = None 
    running_number_added = False

    def __init__(self, data, *args, **kwargs) -> None:    
        self.name = kwargs.pop('name', self.name) 
        self.timestamp = kwargs.pop('timestamp', self.timestamp)
        self.traj_id = kwargs.pop('traj_id', self.traj_id)
        self.mover_id = kwargs.pop('mover_id', self.mover_id)
        self.crs = kwargs.pop('crs', self.crs)   

        if type(data) == str: 
            df = self.load_from_path(data, *args, **kwargs)
        else:
            df = data

        df.rename(columns={self.timestamp: TIMESTAMP}, inplace=True)
        df.rename(columns={self.traj_id: TRAJ_ID}, inplace=True)
        if self.mover_id:
            if self.traj_id == self.mover_id:
                df[MOVER_ID] = df[TRAJ_ID]
            else:
                df.rename(columns={self.mover_id: MOVER_ID}, inplace=True)

        self.df = df

    def load_from_path(self, path, *args, **kwargs):
        if not exists(path):
            msg = f"Error reading from {path}:\r\n"
            msg = f"{msg}Please specify the path to the '{self.name}' file {self.file_name}"
            if self.source_url:
                msg = f"{msg}\r\nYou may download this dataset from: {self.source_url}"
            raise ValueError(msg)
        
        ext = splitext(path)[1]
        if ext == ".pickle":
            nrows = kwargs.pop("nrows", None)
            df = pd.read_pickle(path, *args, **kwargs)
            if nrows: 
                df = df.head(nrows)
        elif ext == ".csv":
            df = pd.read_csv(path, *args, **kwargs)
        elif ext == ".feather":
            df = gpd.read_feather(path, *args, **kwargs)
        elif ext == ".zip":
            df = self.load_df_from_zip_archive(path)
        else:
            df = gpd.read_file(path, *args, **kwargs)
        return df

    def copy(self):
        return Dataset(self.df.copy(), name=self.name, traj_id=self.traj_id, mover_id=self.mover_id, crs=self.crs)
    
    def load_df_from_zip_archive(self, path) -> pd.DataFrame:
        def load_single_csv(csv_name) -> pd.DataFrame:
            print(f"{datetime.now()} Loading {csv_name} ...")
            tmp_df = pd.read_csv(ZipFile(path).open(csv_name)) 
            return tmp_df
         
        df = pd.concat(
            [load_single_csv(csv_name) for csv_name in ZipFile(path).namelist()],
            ignore_index=True
        )
        return df
    
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
        return pd.DataFrame(df)

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

    def to_feather(self, out_path) -> None:
        self.to_gdf().to_feather(out_path)

    def plot(self, *args, **kwargs):
        title = kwargs.pop('title', None)
        df = self.to_df()
        ax = df.plot.scatter(x='x', y='y', *args, **kwargs)
        if title:
            ax.set_title(title)
        return ax
    
    def hvplot(self, *args, **kwargs):
        from hvplot import pandas
        from holoviews import opts
        from holoviews.element import tiles
        opts.defaults(opts.Overlay(active_tools=['wheel_zoom']))
        BG_TILES = tiles.CartoLight()
        gdf = self.to_gdf()
        return BG_TILES * gdf.hvplot(geo=True, *args, **kwargs)

    def datashade(self, *args, **kwargs):
        import datashader as ds
        from hvplot import pandas
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




