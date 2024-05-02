import geopandas as gpd
import movingpandas as mpd
from datetime import timedelta
from mobiml.datasets._dataset import Dataset, TIMESTAMP, SPEED, TRAJ_ID


class AISTripExtractor():
    def __init__(self, gdf: gpd.GeoDataFrame) -> None:
        self.gdf = gdf
        pass

    def get_trips(self, stop_duration=timedelta(minutes=15)) -> mpd.TrajectoryCollection:
        df = self.gdf
        print(f"Original Dataframe size: {len(df)} rows")
        df = df[df[SPEED]>0]
        print(f"   Reduced to: {len(df)} rows after removing records with speed=0")
        print("Creating TrajectoryCollection ...")
        tc = mpd.TrajectoryCollection(df, TRAJ_ID, t=TIMESTAMP, min_length=100)
        print(f"   Created: {tc}")
        print("Generalizing ...")
        tc = mpd.MinTimeDeltaGeneralizer(tc).generalize(tolerance=timedelta(minutes=1))
        print("Splitting ...")
        tc = mpd.ObservationGapSplitter(tc).split(gap=stop_duration)
        print(f"   Split: {tc}")
        return tc 
    
