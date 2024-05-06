from mobiml.datasets._dataset import Dataset, TIMESTAMP, SPEED, TRAJ_ID, MOVER_ID
from .utils import trajectorycollection_to_df

class TrajectoryEnricher():
    def __init__(self, data:Dataset) -> None:
        self.data = data

    def add_speed(self, **kwargs) -> Dataset:
        trajs = self.data.to_trajs()
        trajs.add_speed(**kwargs)
        df = trajectorycollection_to_df(trajs)
        self.data.df = df
        return self.data

    def add_direction(self, **kwargs) -> Dataset:
        trajs = self.data.to_trajs()
        trajs.add_direction(**kwargs)
        df = trajectorycollection_to_df(trajs)
        self.data.df = df
        return self.data


