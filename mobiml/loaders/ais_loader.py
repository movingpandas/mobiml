import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer

from .mover_splitter import MoverSplitter
from mobiml.datasets._dataset import MOVER_ID
from mobiml.datasets.aisdk import SHIPTYPE


def _h3_seq_to_onehot(df, h3_cell_list, h3_seq_col='H3_seq'):
    mlb = MultiLabelBinarizer(sparse_output=True, classes=h3_cell_list)
    h3_onehot = mlb.fit_transform(df.pop(h3_seq_col))
    h3_onehot_df = pd.DataFrame.sparse.from_spmatrix(h3_onehot, index=df.index, columns=mlb.classes_)
    df = df.join(h3_onehot_df)
    return df, mlb.classes_


class AISLoader():
    def __init__(self, vessel_types, traj_features, test_size, path="data/prepared/trajs-stationary.pickle"):
        self.path = path
        self.vessel_types = vessel_types
        self.traj_features = traj_features
        self.test_size = test_size

    def load(self, client_id=None) -> tuple:
        """ 
        Returns train/test values based on pickled trajectories and vessels
        If client_id is set, the trajectories are filtered by the client id 
        """
        if client_id:
            print(f'Client id: {client_id}')

        print(f'Vessel types: {self.vessel_types}')
        print(f'Trajectory features: {self.traj_features}')
        print(f'Test size: {self.test_size}')

        filter = {SHIPTYPE: self.vessel_types}
        if client_id:
            filter['client'] = int(client_id)

        trajs = pd.read_pickle(self.path)
        trajs = self.filter_trajs(filter, trajs)

        if 'H3_seq' in self.traj_features:
            self.traj_features, trajs = self.unstack_h3_seq(self.traj_features, trajs)

        self.min_max_normalize_features(self.traj_features, trajs)
        print(f'Available trajectory columns: {trajs.columns}')

        splitter = MoverSplitter(trajs, mover_id=MOVER_ID, mover_class=SHIPTYPE)
        X_train, X_test, y_train, y_test = splitter.split(self.test_size, self.traj_features, label_col=SHIPTYPE)

        return (X_train.values, y_train.values), (X_test.values, y_test.values)
    
    def min_max_normalize_features(self, features, trajs) -> None:
        trajs[features]=(trajs[features]-trajs[features].min())/(trajs[features].max()-trajs[features].min())

    def unstack_h3_seq(self, features, trajs):
        h3_cell_list = trajs.H3_seq.explode().unique()
        trajs, onehot_cols = _h3_seq_to_onehot(trajs, h3_cell_list)
        features.remove('H3_seq')
        features = features + list(onehot_cols)
        return features, trajs

    def filter_trajs(self, filter, trajs) -> list:
        if filter:
            for key, value in filter.items():
                print(f'Filtering {key} to {value} ...')
                if type(value)==list:
                    trajs = trajs[trajs[key].isin(value)]
                else:
                    trajs = trajs[trajs[key]==value]
                print(f'... {len(trajs)} found.')
        return trajs
   