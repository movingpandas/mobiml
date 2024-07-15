from mobiml.utils import LogRegParams


import numpy as np
from sklearn.linear_model import LogisticRegression


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
