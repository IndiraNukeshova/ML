import typing
import numpy as np


class MinMaxScaler:
    def fit(self, data):
        """Store calculated statistics
        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.mins = np.amin(data, axis=0)
        self.maxs = np.amax(data, axis=0)

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        return (data - self.mins) / (self.maxs - self.mins)


class StandardScaler:
    def fit(self, data):
        """Store calculated statistics
        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.expectation = np.mean(data, axis=0)
        self.deviance = np.std(data, axis=0)

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        return (data - self.expectation) / self.deviance
