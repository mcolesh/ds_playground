"""Implementation of the perceptron classifier algorithm"""

import numpy as np


class Perceptron:
    """The perceptron classifier algorithm"""

    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000) -> None:
        """Initializing Perceptron with learning-rate and number of iterations over the testing-data"""
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y) -> None:
        """Fitting the data (learning...🤔)"""
        n_samples, n_features = X.shape

        # init weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # converting targets to 1s and 0s
        y_ = np.array([1 if i > 0 else 0 for i in y])

        # number of iterations over the samples (training-set)
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                
                # The optimization procedure (gradient decent)
                # notice-1: y_predicted == y => update = 0 (weights say the same)
                # notice-2:  y_predicted != y => update * xi is the (with direction) step size 
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X_test) -> np.ndarray:
        """
            returns a prediction for a given testing set
            output is of the shape [1,0,1,0,1]
        """
        linear_output = np.dot(X_test, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x: float) -> np.ndarray:
        """
            The activation function
            output is of the form [1] or [0]
        """
        return np.where(x >= 0, 1, 0)
