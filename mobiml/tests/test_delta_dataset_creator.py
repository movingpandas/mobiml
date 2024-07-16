import os

from mobiml.loaders.temporal_splitter import TemporalSplitter

from mobiml.datasets.brest_ais import PreprocessedBrestAIS

from mobiml.transforms.delta_dataset_creator import DeltaDatasetCreator


class TestDeltaDatasetCreator:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def test_delta_dataset(self):
        path = os.path.join(
            self.test_dir, "data/test_nautilus_trajectories_preprocessed_100.csv"
        )
        dataset = PreprocessedBrestAIS(path)
        ais = TemporalSplitter(dataset).split()
        delta_dataset = DeltaDatasetCreator(ais)
        assert isinstance(delta_dataset, DeltaDatasetCreator)
        traj_delta = delta_dataset.get_delta_dataset("split", njobs=1)

        expected_dt_curr = [259, 80, 91]
        dt_curr = traj_delta.dt_curr.tolist()
        assert dt_curr[:3] == expected_dt_curr

        traj_delta_windows = delta_dataset.get_windowed_dataset("split")
        assert len(traj_delta_windows) == 1

        expected_samples = [259, 80, 91]
        samples_list = traj_delta_windows.samples.tolist()
        assert samples_list[0][0][2] == expected_samples[0]
        assert samples_list[0][1][2] == expected_samples[1]
        assert samples_list[0][2][2] == expected_samples[2]
        assert len(samples_list[0]) == 67

        expected_labels = [81.82, -26.76]
        labels_list = traj_delta_windows.labels.tolist()
        assert round(labels_list[0][0], 2) == expected_labels[0]
        assert round(labels_list[0][1], 2) == expected_labels[1]
        assert len(labels_list[0]) == 2
