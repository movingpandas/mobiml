import os
import pytest
from movingpandas import TrajectoryCollection

from mobiml.datasets import (
    BrestAIS,
    PreprocessedBrestAIS,
    SPEED,
    DIRECTION,
)


class TestBrestAIS:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def test_data_from_csv(self):
        path = os.path.join(self.test_dir, "data", "test_nari_dynamic.csv")
        data = BrestAIS(path)
        assert isinstance(data, BrestAIS)
        assert SPEED in data.df.columns
        assert DIRECTION in data.df.columns
        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)
        assert len(trajs) > 0
        assert len(data.df) == 10
        assert data.df["x"].iloc[0] == pytest.approx(-4.4657183)
        assert data.df["y"].iloc[0] == pytest.approx(48.38249)


class TestPreprocessedBrestAIS:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def test_data_from_csv(self):
        path = os.path.join(
            self.test_dir, "data", "test_nautilus_trajectories_preprocessed.csv"
        )
        data = PreprocessedBrestAIS(path)
        assert isinstance(data, PreprocessedBrestAIS)
        assert SPEED in data.df.columns
        assert DIRECTION in data.df.columns
        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)
        assert len(trajs) > 0
        assert len(data.df) == 10
        assert data.df["x"].iloc[0] == pytest.approx(-5.33829)
        assert data.df["y"].iloc[0] == pytest.approx(48.2961)
