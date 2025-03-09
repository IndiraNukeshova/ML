import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype
        self.new_features_amount = 0
        self.unique_values = []

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        arr = np.array(X)
        for col in X:
            if X[col].dtype == object:
                self.unique_values.append(np.unique(np.array(X[col], dtype=str)))
            else:
                self.unique_values.append(np.unique(np.array(X[col])))
        for i in range(len(self.unique_values)):
            self.new_features_amount += self.unique_values[i].shape[0]

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        X = np.array(X)
        ret = np.zeros((X.shape[0], self.new_features_amount))
        assert X.shape[1] == len(self.unique_values), "incorrect sizes"
        current_pos = 0
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                current_shift = self.unique_values[j].tolist().index(X[i, j])
                ret[i, current_pos + current_shift] = 1
                current_pos += self.unique_values[j].shape[0]
            current_pos = 0
        return ret

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        self.statistics = []
        self.unique_values = []

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        assert len(Y.shape) == 1, "Y should be 1-dimensional"
        arr = np.array(X)
        assert arr.shape == X.shape, "u r wrong"
        self.statistics = [[] for _ in range(arr.shape[1])]
        for i in range(arr.shape[1]):
            current = arr[:, i]
            self.unique_values.append(np.unique(current))
            for j in self.unique_values[-1]:
                positions = np.where(current == j)[0]
                successes = Y[positions].sum() / positions.shape[0]
                counters = positions.shape[0] / X.shape[0]
                self.statistics[i].append((successes, counters))

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        arr = X.values
        ret = np.zeros((arr.shape[0], arr.shape[1] * 3), dtype=np.float32)
        for j in range(arr.shape[1]):
            for i in range(arr.shape[0]):
                cur_value = arr[i][j]
                current_unique_values = self.unique_values[j]
                current_stat = self.statistics[j][np.where(current_unique_values == cur_value)[0][0]]
                ret[i][3 * j], ret[i][3 * j + 1], ret[i][3 * j + 2] = (
                    current_stat[0],
                    current_stat[1],
                    (current_stat[0] + a) / (current_stat[1] + b)
                )
        return ret

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds
        self.partition = []
        self.new_X = None

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        for i in group_k_fold(X.shape[0], self.n_folds, seed):
            self.partition.append(i)
        self.new_X = np.zeros((X.shape[0], X.shape[1] * 2), dtype=np.float64)
        assert len(Y.shape) == 1, "Y should be 1-dimensional"
        arr = X.values
        for fold in self.partition:
            for col in range(arr.shape[1]):
                except_fold = arr[fold[0], col]
                data = arr[fold[1], col]
                unique_values = np.unique(except_fold)
                for val in unique_values:
                    positions = np.where(data == val)[0]
                    amount = positions.shape[0]
                    target = Y.values[fold[1]][positions].sum()
                    self.new_X[fold[0][np.where(except_fold == val)[0]], col*2:col*2+2] = (
                        target / amount,
                        amount / fold[1].shape[0]
                    )

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        res = np.zeros((self.new_X.shape[0], 0), dtype=np.float64)
        for i in range(0, self.new_X.shape[1], 2):
            res = np.hstack((
                res,
                self.new_X[:, i].reshape(-1, 1),
                self.new_X[:, i + 1].reshape(-1, 1),
                ((self.new_X[:, i] + a) / (self.new_X[:, i + 1] + b)).reshape(-1, 1)
            ))
        return res

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    iterations = 1000
    learning_rate = 1e-2
    x_enc = np.eye(np.unique(x).shape[0])[x]
    w = np.zeros(x_enc.shape[1])
    for i in range(iterations):
        p = x_enc @ w
        grad = x_enc.T @ (p - y)
        w -= learning_rate * grad
    return w
