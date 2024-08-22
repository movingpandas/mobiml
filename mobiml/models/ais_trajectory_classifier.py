import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy
from typing import Dict, List, Tuple
from pathlib import Path

import flwr as fl
from flwr.common import Metrics

from sklearn.metrics import log_loss
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression

from mobiml.transforms import MoverSplitter
from mobiml.datasets import MOVER_ID, SHIPTYPE
from mobiml.utils import LogRegParams


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model, data_loader, scenario_name):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = data_loader.load()

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        model.set_model_params(parameters)
        vessel_types = model.classes

        # deep copy to calcuate accuracy using real class names.
        # (FL aggregation for class names does not work since only numpy values are supported. )
        model2 = deepcopy(model)
        model2.classes_ = vessel_types

        predicted = model.predict_proba(X_test)
        loss = log_loss(y_test, predicted, labels=vessel_types)
        accuracy = model2.score(X_test, y_test)
        predictions = model2.predict(X_test)
        # ml_utils.save_metrics(y_test, predictions, scenario_name)
        # ml_utils.display_confusion_matrix(y_test, predictions, vessel_types)
        print("Accuracy", accuracy)
        return loss, {"accuracy": accuracy}

    return evaluate


def weighted_average(metrics: list[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def _h3_seq_to_onehot(df, h3_cell_list, h3_seq_col="H3_seq"):
    mlb = MultiLabelBinarizer(sparse_output=True, classes=h3_cell_list)
    h3_onehot = mlb.fit_transform(df.pop(h3_seq_col))
    h3_onehot_df = pd.DataFrame.sparse.from_spmatrix(
        h3_onehot, index=df.index, columns=mlb.classes_
    )
    df = df.join(h3_onehot_df)
    return df, mlb.classes_


class AISLoader:
    def __init__(
        self,
        vessel_types,
        traj_features,
        test_size,
        path="data/prepared/trajs-stationary.pickle",
    ):
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
            print(f"Client id: {client_id}")

        print(f"Vessel types: {self.vessel_types}")
        print(f"Trajectory features: {self.traj_features}")
        print(f"Test size: {self.test_size}")

        filter = {SHIPTYPE: self.vessel_types}
        if client_id:
            filter["client"] = int(client_id)

        trajs = pd.read_pickle(self.path)
        trajs = self.filter_trajs(filter, trajs)

        if "H3_seq" in self.traj_features:
            self.traj_features, trajs = self.unstack_h3_seq(self.traj_features, trajs)

        self.min_max_normalize_features(self.traj_features, trajs)
        print(f"Available trajectory columns: {trajs.columns}")

        splitter = MoverSplitter(trajs, mover_id=MOVER_ID, mover_class=SHIPTYPE)
        X_train, X_test, y_train, y_test = splitter.split(
            self.test_size, self.traj_features, label_col=SHIPTYPE
        )

        return (X_train.values, y_train.values), (X_test.values, y_test.values)

    def min_max_normalize_features(self, features, trajs) -> None:
        trajs[features] = (trajs[features] - trajs[features].min()) / (
            trajs[features].max() - trajs[features].min()
        )

    def unstack_h3_seq(self, features, trajs):
        h3_cell_list = trajs.H3_seq.explode().unique()
        trajs, onehot_cols = _h3_seq_to_onehot(trajs, h3_cell_list)
        features.remove("H3_seq")
        features = features + list(onehot_cols)
        return features, trajs

    def filter_trajs(self, filter, trajs) -> list:
        if filter:
            for key, value in filter.items():
                print(f"Filtering {key} to {value} ...")
                if type(value) == list:
                    trajs = trajs[trajs[key].isin(value)]
                else:
                    trajs = trajs[trajs[key] == value]
                print(f"... {len(trajs)} found.")
        return trajs


class SummarizedAISTrajectoryClassifier(LogisticRegression):
    def __init__(self, classes, n_features) -> None:
        super().__init__(
            penalty="l2",
            max_iter=1,  # local epoch
            warm_start=True,  # prevent refreshing weights when fitting
            multi_class="multinomial",
        )
        self.classes = classes
        self.n_classes = len(classes)
        self.n_features = n_features
        # Setting initial parameters, akin to model.compile for keras models
        self.set_initial_params()

    def set_initial_params(self):
        """Sets initial parameters as zeros Required since model params are
        uninitialized until model.fit is called.
        But server asks for initial parameters from clients at launch. Refer
        to sklearn.linear_model.LogisticRegression documentation for more
        information.
        """
        self.classes_ = np.array([i for i in range(self.n_classes)])
        self.coef_ = np.zeros((self.n_classes, self.n_features))
        self.coef_ = np.zeros((self.n_classes, self.n_features))

        if self.fit_intercept:
            self.intercept_ = np.zeros((self.n_classes,))

    def set_model_params(self, params: LogRegParams):
        """Sets the parameters of a sklean LogisticRegression model."""
        self.coef_ = params[0]
        # model.classes_ = params[1]

        if self.fit_intercept:
            self.intercept_ = params[1]

    def get_model_parameters(self) -> LogRegParams:
        """Returns the paramters of a sklearn LogisticRegression model."""
        if self.fit_intercept:
            params = [
                self.coef_,
                # model.classes_,
                self.intercept_,
            ]
        else:
            params = [
                self.coef_,
                # model.classes_,
            ]
        return params
