from movingpandas import ObservationGapSplitter, TemporalSplitter
from mobiml.datasets import Dataset
from .utils import trajectorycollection_to_df


class TrajectorySplitter:
    def __init__(self, data: Dataset) -> None:
        self.data = data

    def split(self, **kwargs) -> Dataset:
        """
        Split trajectories by different rules.

        Parameters
        ----------
        observation_gap : datetime.timedelta
            Time gap threshold.
        temporal_split_mode : string
            Split mode. ('hour', 'day', 'month' or 'year')

        Examples
        --------

        >>> TrajectorySlitter(dataset).split(observation_gap=timedelta(hours=1))

        >>> TrajectorySlitter(dataset).split(temporal_split_mode='day')
        """

        split_by_observation_gap = kwargs.pop("observation_gap", None)
        split_temporally = kwargs.pop("temporal_split_mode", None)

        trajs = self.data.to_trajs()

        if split_by_observation_gap:
            trajs = ObservationGapSplitter(trajs).split(gap=split_by_observation_gap)
        if split_temporally:
            trajs = TemporalSplitter(trajs).split(mode=split_temporally)

        df = trajectorycollection_to_df(trajs)
        self.data.df = df
        return self.data
