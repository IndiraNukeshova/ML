import typing
import numpy as np
from collections import defaultdict


def kfold_split(num_objects, num_folds):
    all_ind = np.arange(num_objects)
    step = num_objects // num_folds
    res_ind = []
    for fold in range(num_folds - 1):
        left_ind = np.concatenate((all_ind[:step * fold], all_ind[step * (fold + 1):]), dtype=np.int32, casting='unsafe')
        right_ind = all_ind[step * fold:step * (fold + 1)]
        res_ind.append((left_ind, right_ind))
    left_ind = all_ind[:step * (num_folds - 1)]
    right_ind = all_ind[step * (num_folds - 1):]
    res_ind.append((left_ind, right_ind))
    return res_ind


def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    result = dict()
    for normalizers in parameters['normalizers']:
        for neighbors in parameters['n_neighbors']:
            for metrics in parameters['metrics']:
                for weights in parameters['weights']:
                    mean_val = np.empty(len(folds))
                    for step in range(len(folds)):
                        model = knn_class(n_neighbors=neighbors, weights=weights, metric=metrics)
                        if normalizers[0] is None:
                            X_train = X[folds[step][0]]
                            X_test = X[folds[step][1]]
                        else:
                            scaler = normalizers[0]
                            scaler.fit(X[folds[step][0]])
                            X_train = scaler.transform(X[folds[step][0]])
                            X_test = scaler.transform(X[folds[step][1]])
                        model.fit(X_train, y[folds[step][0]])
                        y_predict = model.predict(X_test)
                        mean_val[step] = score_function(y[folds[step][1]], y_predict)
                    result[(normalizers[1], neighbors, metrics, weights)] = np.mean(mean_val)
    return result
