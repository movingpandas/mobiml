import sys 
import flwr as fl
import numpy as np
from copy import deepcopy
from typing import Dict, List, Tuple
from pathlib import Path
from sklearn.metrics import log_loss
from flwr.common import Metrics

import utils
import ml_utils 
from mobiml.loaders.ais_loader import AISLoader
from mobiml.models import SummarizedAISTrajectoryClassifier 


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: SummarizedAISTrajectoryClassifier, data_loader, scenario_name):
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
        ml_utils.save_metrics(y_test, predictions, scenario_name)
        ml_utils.display_confusion_matrix(y_test, predictions, vessel_types)
        print("Accuracy", accuracy)
        return loss, {"accuracy": accuracy}
    
    return evaluate


def weighted_average(metrics: list[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def main():
    # Start Flower server for federated learning
    np.random.seed(0)

    try:
        data_path = sys.argv[1]
    except IndexError:
        data_path = "data/prepared/training-data-stationary.pickle"
    scenario_name =  Path(data_path).stem.replace('training-data-','')

    utils.print_logo()
    vessel_types, n_features, traj_features, test_size = utils.get_dvc_params()

    data_loader = AISLoader(vessel_types, traj_features, test_size, path=data_path)

    model = SummarizedAISTrajectoryClassifier(vessel_types, n_features)
    
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model, data_loader, scenario_name),
        on_fit_config_fn=fit_round,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )

    print(f"""
    =====================
       Starting Server
    =====================
    """)

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=10),
    )


if __name__ == "__main__":
    main()
