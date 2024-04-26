import sys
import time
import warnings
import flwr as fl
import numpy as np
from sklearn.metrics import log_loss
import ml_utils

import utils
import ml_utils
from mobiml.models import SummarizedAISTrajectoryClassifier
from mobiml.loaders import AISLoader


class AisClient(fl.client.NumPyClient):
    """ Client for FL model """
    def get_parameters(self, config):  # type: ignore
        return model.get_model_parameters()

    def fit(self, parameters, config):  # type: ignore
        model.set_model_params(parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
            accuracy = model.score(X_train, y_train)

        print(f"Training finished for round {config['server_round']}")
        return model.get_model_parameters(), len(X_train), {"accuracy": accuracy}

    def evaluate(self, parameters, config):  # type: ignore
        model.set_model_params(parameters)
        vessel_types = model.classes
        loss = log_loss(y_test, model.predict_proba(X_test), labels = vessel_types)
        accuracy = model.score(X_test, y_test)
        return loss, len(X_test), {"accuracy": accuracy}
    
    
if __name__ == "__main__":
    np.random.seed(0)
    client_id = sys.argv[1]

    try:
        data_path = sys.argv[2]
    except IndexError:
        data_path = "data/prepared/trajs-stationary.pickle"

    vessel_types, n_features, traj_features, test_size = utils.get_dvc_params()

    print(f"""
    ===========================
       Loading Client {client_id} Data
    ===========================
    """)

    data_loader = AISLoader(vessel_types, traj_features, test_size, path=data_path)
    (X_train, y_train), (X_test, y_test) = data_loader.load(client_id=client_id)

    # Split train set into partitions and randomly use one for training.
    n_clients = 3
    partition_id = np.random.choice(n_clients)
    (X_train, y_train) = ml_utils.partition(X_train, y_train, n_clients)[partition_id]

    model = SummarizedAISTrajectoryClassifier(vessel_types, n_features)

    # Wait to try to ensure that the server is ready before the clients connect
    time.sleep(3)  
    
    print(f"""
    =======================
       Starting Client {client_id}
    =======================
    """)

    # Start Flower client
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=AisClient())
