from movingpandas.unit_utils import UNITS

from mobiml.datasets import _Dataset
from .utils import trajectorycollection_to_df


class TrajectoryEnricher:
    def __init__(self, data: _Dataset) -> None:
        self.data = data

    def add_speed(self, **kwargs) -> _Dataset:
        print("Adding speed ...")
        trajs = self.data.to_trajs()
        trajs.add_speed(**kwargs)
        df = trajectorycollection_to_df(trajs)
        self.data.df = df
        return self.data

    def add_direction(self, **kwargs) -> _Dataset:
        print("Adding direction")
        trajs = self.data.to_trajs()
        trajs.add_direction(**kwargs)
        df = trajectorycollection_to_df(trajs)
        self.data.df = df
        return self.data

    def add_features(self, n_threads=5, **kwargs) -> _Dataset:
        speed = kwargs.pop("speed", False)
        direction = kwargs.pop("direction", False)
        speed_units = kwargs.pop("speed_units", UNITS())
        acceleration = kwargs.pop("acceleration", False)
        acceleration_units = kwargs.pop("acceleration_units", UNITS())
        overwrite = kwargs.pop("overwrite", False)
        print("Preparing trajectories ...")
        trajs = self.data.to_trajs()
        if speed:
            print("Adding speed ...")
            trajs.add_speed(units=speed_units, overwrite=overwrite, n_threads=n_threads)
        if direction:
            print("Adding direction ...")
            trajs.add_direction(overwrite=overwrite, n_threads=n_threads)
        if acceleration:
            print("Adding acceleration ...")
            trajs.add_acceleration(
                units=acceleration_units,
                overwrite=overwrite,
                n_threads=n_threads,
            )
        df = trajectorycollection_to_df(trajs)
        self.data.df = df
        return self.data
