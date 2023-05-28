from datetime import timedelta
from .datasets import Dataset, TIMESTAMP, SPEED, TRAJ_ID
import movingpandas as mpd

class AISTripExtractor():
    def __init__(self, data: Dataset) -> None:
        self.data = data
        pass

    def get_trips(self, stop_duration=timedelta(minutes=15)) -> mpd.TrajectoryCollection:
        df = self.data.to_gdf()
        print(f"Original Dataframe size: {len(df)} rows")
        df = df[df[SPEED]>0]
        print(f"Reduced to: {len(df)} rows after removing records with speed=0")
        tc = mpd.TrajectoryCollection(df, TRAJ_ID, t=TIMESTAMP, min_length=100)
        print(f"Created: {tc}")
        print("Generalizing ...")
        tc = mpd.MinTimeDeltaGeneralizer(tc).generalize(tolerance=timedelta(minutes=1))
        print("Splitting ...")
        tc = mpd.ObservationGapSplitter(tc).split(gap=stop_duration)
        print(f"Extracted: {tc}")
        return tc 
    
