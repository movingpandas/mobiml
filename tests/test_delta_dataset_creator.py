import os

from mobiml.datasets import PreprocessedBrestAIS
from mobiml.transforms import DeltaDatasetCreator


class TestDeltaDatasetCreator:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def setup_method(self):
        path = os.path.join(
            self.test_dir,
            "data",
            "test_nautilus_trajectories_preprocessed_100.csv",
        )
        dataset = PreprocessedBrestAIS(path)
        self.delta_dataset_creator = DeltaDatasetCreator(dataset)

    def test_get_delta_dataset(self):
        delta_dataset = self.delta_dataset_creator.get_delta_dataset(njobs=1)

        expected_dt_curr = [259, 80, 91]
        dt_curr = delta_dataset.dt_curr.tolist()
        assert dt_curr[:3] == expected_dt_curr

    def test_get_windowed_dataset(self):
        windowed_dataset = self.delta_dataset_creator.get_windowed_dataset()
        assert len(windowed_dataset) == 1

        samples_list = windowed_dataset.samples.tolist()
        assert samples_list[0][0][2] == 259
        assert samples_list[0][1][2] == 80
        assert samples_list[0][2][2] == 91

    def test_windowing(self):
        delta_dataset = self.delta_dataset_creator.get_delta_dataset(njobs=1)
        samples, labels = self.delta_dataset_creator.traj_windowing(
            delta_dataset, 10, 2, 30
        )
        assert len(samples) == len(labels) == 4
        samples, labels = self.delta_dataset_creator.traj_windowing(
            delta_dataset, 10, 2, 100
        )
        assert len(samples) == len(labels) == 1
        samples, labels = self.delta_dataset_creator.traj_windowing(
            delta_dataset, 10, 2, 10
        )
        assert len(samples) == len(labels) == 10
