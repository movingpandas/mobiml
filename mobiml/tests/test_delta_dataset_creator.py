import os

from mobiml.datasets import PreprocessedBrestAIS

from mobiml.transforms import DeltaDatasetCreator, TemporalSplitter


class TestDeltaDatasetCreator:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def test_get_delta_dataset(self):
        path = os.path.join(
            self.test_dir, "data/test_nautilus_trajectories_preprocessed_100.csv"
        )
        dataset = PreprocessedBrestAIS(path)
        split_dataset = TemporalSplitter(dataset).split(dev_size=0.25, test_size=0.25) # TODO: refactor so that DeltaDatasetCreator is not dependent on TemporalSplitter
        delta_dataset_creator = DeltaDatasetCreator(split_dataset)
        delta_dataset = delta_dataset_creator.get_delta_dataset("split", njobs=1)

        expected_dt_curr = [259, 80, 91]
        dt_curr = delta_dataset.dt_curr.tolist()
        assert dt_curr[:3] == expected_dt_curr

    def test_get_windowed_dataset(self):
        path = os.path.join(
            self.test_dir, "data/test_nautilus_trajectories_preprocessed_100.csv"
        )
        dataset = PreprocessedBrestAIS(path)
        split_dataset = TemporalSplitter(dataset).split(dev_size=0.25, test_size=0.25)
        delta_dataset_creator = DeltaDatasetCreator(split_dataset)
        windowed_dataset = delta_dataset_creator.get_windowed_dataset("split")
        assert len(windowed_dataset) == 1

        samples_list = windowed_dataset.samples.tolist()
        assert samples_list[0][0][2] == 259
        assert samples_list[0][1][2] == 80
        assert samples_list[0][2][2] == 91

    def test_windowing(self):
        path = os.path.join(
            self.test_dir, "data/test_nautilus_trajectories_preprocessed_100.csv"
        )
        dataset = PreprocessedBrestAIS(path)
        split_dataset = TemporalSplitter(dataset).split(dev_size=0.25, test_size=0.25)
        delta_dataset_creator = DeltaDatasetCreator(split_dataset)
        delta_dataset = delta_dataset_creator.get_delta_dataset("split", njobs=1)
        samples, labels = delta_dataset_creator.traj_windowing(delta_dataset, 10, 2, 30)
        assert len(samples) == len(labels) == 4
        samples, labels = delta_dataset_creator.traj_windowing(delta_dataset, 10, 2, 100)
        assert len(samples) == len(labels) == 1
        samples, labels = delta_dataset_creator.traj_windowing(delta_dataset, 10, 2, 10)
        assert len(samples) == len(labels) == 10

