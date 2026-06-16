import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# we want a scaler that removes a lot of intermediate data.
# we take the max of each sample, threshold with some percentage
# Then do the L2 scaler on the result


class PCAScaler:
    def __init__(self, n_comps):
        self.n_comps = n_comps
        self.pca = PCA(self.n_comps)

    def fit(self, X):
        # [b, c]
        self.pca.fit(self._get_normed_raw_data(X))
        return self

    def transform(self, X):
        return self.pca.transform(self._get_normed_raw_data(X))

    def _get_normed_raw_data(self, X):
        X = normalize(X, "l2")
        return X - X.mean(axis=0)


class SparseL2Scaler:
    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def fit(self, X):
        return self

    def transform(self, X):
        maxes = np.abs(X).max(axis=0)
        threshes = maxes * self.threshold
        X[np.abs(X) < threshes] = 0
        return normalize(X, "l2")


class L2Scaler:
    def fit(self, X):
        return self

    def transform(self, X):
        return normalize(X, "l2")

    def inverse_transform(self, X):
        raise NotImplementedError


class MaxScaler:
    def fit(self, X):
        return self

    def transform(self, X):
        return normalize(X, "max")

    def inverse_transform(self, X):
        raise NotImplementedError


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
