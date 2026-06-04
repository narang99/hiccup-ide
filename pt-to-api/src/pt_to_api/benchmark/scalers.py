import numpy as np


class NormaliseStdScaler:
    def __init__(self):
        self.global_std_ = None

    def fit(self, X):
        X = np.array(X)
        self.global_std_ = X.std()
        return self

    def transform(self, X):
        X = np.array(X)
        return X / self.global_std_

    def inverse_transform(self, X):
        X = np.array(X)
        return X * self.global_std_


class MeanPerDimGlobalStdScaler:
    def __init__(self):
        self.means_ = None
        self.global_std_ = None

    def fit(self, X):
        X = np.array(X)
        self.means_ = X.mean(axis=0)
        self.global_std_ = (X - self.means_).std()  # std of demeaned X
        return self

    def transform(self, X):
        X = np.array(X)
        return (X - self.means_) / self.global_std_

    def inverse_transform(self, X):
        X = np.array(X)
        return (X * self.global_std_) + self.means_
