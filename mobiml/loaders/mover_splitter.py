import pandas as pd 
from datetime import datetime
from sklearn.model_selection import train_test_split

from mobiml.datasets.aisdk import SHIPTYPE

class MoverSplitter:
    def __init__(self, trajs, mover_id, mover_class) -> None:
        self.trajs = trajs
        self.mover_id = mover_id
        self.mover_class = mover_class 
        self.movers = self.get_labelled_mover_list()  

    def split(self, test_size, features, label_col=SHIPTYPE):
        X_cols = features
        y_col = label_col

        print(f"{datetime.now()} Splitting dataset ...")
        X = self.movers[self.mover_id]
        y = self.movers[label_col]
        movers_train, movers_test, _, _ = train_test_split(
            X, y, test_size=test_size, random_state=0, stratify=self.movers[self.mover_class])
        
        print(f"Using {len(movers_train.index)} movers for training and {len(movers_test.index)} for testing ...")

        tmp = self.trajs.sample(frac=1).reset_index(drop=True)

        trajs_train = tmp[tmp[self.mover_id].isin(movers_train)]
        trajs_test = tmp[tmp[self.mover_id].isin(movers_test)]
        print(f"({len(trajs_train.index)} trajectories for training and {len(trajs_test.index)} for testing)")

        X_train = trajs_train[X_cols]
        X_test = trajs_test[X_cols]
        y_train = trajs_train[y_col]
        y_test = trajs_test[y_col]

        return X_train, X_test, y_train, y_test

    def get_labelled_mover_list(self) -> pd.DataFrame:
        movers = self.trajs.groupby(self.mover_id)[[self.mover_class]].agg(pd.Series.mode)
        movers = movers.reset_index()
        return movers 