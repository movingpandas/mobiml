from movingpandas import ObservationGapSplitter
from mobiml.datasets._dataset import Dataset, TIMESTAMP, SPEED, TRAJ_ID, MOVER_ID
from .utils import trajectorycollection_to_df

class TrajectorySplitter():
    def __init__(self, data:Dataset) -> None:
        self.data = data

    def split(self, **kwargs) -> Dataset:
        observation_gap = kwargs.pop('observation_gap', None)

        trajs = self.data.to_trajs()
        
        if observation_gap:
            trajs = ObservationGapSplitter(trajs).split(gap=observation_gap)

        df = trajectorycollection_to_df(trajs)
        self.data.df = df
        return self.data



