import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from mobiml.utils import XYList


def display_confusion_matrix(y_test, predictions, labels):
    cm = confusion_matrix(y_test, predictions, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)


def save_metrics(predictions, y_test, scenario_name):
    if not os.path.exists("output"):
        os.makedirs("output")

    metrics = {"accuracy": accuracy_score(y_test, predictions)}

    out_path = f"output/fl-global-metrics-{scenario_name}.json"
    print(f"Saving metrics to {out_path}")
    with open(out_path, "w") as fd:
        json.dump(metrics, fd)


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    zipped = zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    return list(zipped)
