import os

from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor

import numpy as np


class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """
    def fit(self, x, y):
        """
        Build the transformer on the training set.
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: trained transformer
        """
        return self

    def fit_transform(self, x, y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        return self.transform(x)

    def transform(self, x):
        """
        Transform the list of potential's 2d matrices with the trained transformer.
        :param x: list of potential's 2d matrices
        :return: transformed potentials (list of 1d vectors)
        """
        centered = []
        for img in x:
            if (np.all((img == 0) | (img == 20.))):
                y, x = np.where(img == np.min(img))
                center_y = y[-1] - (y[-1] - y[0]) // 2
                center_x = x[-1] - (x[-1] - x[0]) // 2
            else:
                center_y, center_x = np.unravel_index(np.argmin(img), img.shape)
            center_shift_y = 128 - center_y
            center_shift_x = 128 - center_x
            new_img = np.zeros((256, 256))
            new_img.fill(20.)
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    if (img[y][x] != 20.):
                        if (0 <= y + center_shift_y <= 255 and 0 <= x + center_shift_x <= 255):
                            new_img[y + center_shift_y][x + center_shift_x] = img[y][x]
            centered.append(new_img)
        centered = np.array(centered)
        return centered.reshape((centered.shape[0], -1))


def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    # it's suggested to modify only the following line of this function
    regressor = Pipeline([('vectorizer', PotentialTransformer()), ('decision_tree', ExtraTreesRegressor(max_features='sqrt', random_state=228))])
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}
