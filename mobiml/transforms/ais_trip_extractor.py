import geopandas as gpd
import movingpandas as mpd
from datetime import timedelta
from mobiml.datasets._dataset import Dataset, TIMESTAMP, SPEED, TRAJ_ID


class AISTripExtractor():
    def __init__(self, data) -> None:
        if isinstance(data, gpd.GeoDataFrame):
            gdf = data
            print(f"Original Dataframe size: {len(gdf)} rows")
            gdf = gdf[gdf[SPEED]>0]
            print(f"   Reduced to: {len(gdf)} rows after removing records with speed=0")
            print("Creating TrajectoryCollection ...")
            self.tc = mpd.TrajectoryCollection(gdf, TRAJ_ID, t=TIMESTAMP, min_length=100)
        elif isinstance(data, mpd.TrajectoryCollection):
            self.tc = data
        else:
            raise TypeError(f"Invalid input data {data}")
        print(f"   Created: {self.tc}")

    def get_trips(self, gap_duration=timedelta(minutes=15)) -> mpd.TrajectoryCollection:
        print("Generalizing ...")
        self.tc = mpd.MinTimeDeltaGeneralizer(self.tc).generalize(tolerance=timedelta(minutes=1))
        print(f"Splitting at observation gaps ({gap_duration}) ...")
        self.tc = mpd.ObservationGapSplitter(self.tc).split(gap=gap_duration)
        print(f"   Split: {self.tc}")
        return self.tc 
    
