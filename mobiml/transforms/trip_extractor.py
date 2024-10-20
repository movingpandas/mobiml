import geopandas as gpd
import movingpandas as mpd
from datetime import timedelta
from mobiml.datasets import Dataset, SPEED, TIMESTAMP, TRAJ_ID


class TripExtractor:
    def __init__(self, data, min_length=100) -> None:
        if isinstance(data, gpd.GeoDataFrame):
            gdf = data
            print(f"Original Dataframe size: {len(gdf)} rows")
            try:
                gdf = gdf[gdf[SPEED] > 0]
            except KeyError:
                pass
            print(f"   Reduced to: {len(gdf)} rows after removing records with speed=0")
            print("Creating TrajectoryCollection ...")
            self.tc = mpd.TrajectoryCollection(
                gdf, TRAJ_ID, t=TIMESTAMP, min_length=min_length
            )
        elif isinstance(data, mpd.TrajectoryCollection):
            self.tc = data
        elif isinstance(data, Dataset):
            self.tc = data.to_trajs()
        else:
            raise TypeError(f"Invalid input data {data}")
        print(f"   Created: {self.tc}")

    def get_trips(
        self,
        gap_duration=timedelta(minutes=15),
        generalization_tolerance=timedelta(minutes=1),
    ) -> mpd.TrajectoryCollection:
        print("Generalizing ...")
        self.tc = mpd.MinTimeDeltaGeneralizer(self.tc).generalize(
            tolerance=generalization_tolerance
        )
        print(f"Splitting at observation gaps ({gap_duration}) ...")
        self.tc = mpd.ObservationGapSplitter(self.tc).split(gap=gap_duration)
        print(f"   Split: {self.tc}")
        return self.tc
