import numpy as np
from typing import Tuple

def sum_non_neg_diag(X: np.ndarray) -> int: 
    v = np.diagonal(X) 
    v = v[np.where(v >= 0)] 
    if not(v.size): 
        return -1 
    return np.sum(v)


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool: 
    if len(x) != len(y): 
        return False 
 
    x_list = x.tolist() 
    y_list = y.tolist() 
 
    x_list.sort() 
    y_list.sort() 
 
    return x_list == y_list



def max_prod_mod_3(x: np.ndarray) -> int:
    x_copy = x.copy()
    
    y = x_copy[1:] * x[:-1]
    y_divisible_by_3 = y[y % 3 == 0]
    if y_divisible_by_3.size == 0:
        return -1
    return np.max(y_divisible_by_3)



def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray: 
    if image.shape[-1] != len(weights): 
        raise ValueError("Количество каналов изображения должно соответствовать количеству весов") 
 
    weighted_channels = image * weights.reshape((1, 1, -1)) 
    result_image = np.sum(weighted_channels, axis=-1) 
 
    return result_image



def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    if np.sum(x[:, 1]) != np.sum(y[:, 1]):
        return -1

    raveled_x = np.repeat(x[:, 0], x[:, 1])
    raveled_y = np.repeat(y[:, 0], y[:, 1])

    return np.sum(raveled_x * raveled_y)



def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    x_lens = np.sum(X * X, axis=1) ** 0.5
    y_lens = np.sum(Y * Y, axis=1) ** 0.5

    M = np.dot(X, Y.T).astype(np.float64)

    M[x_lens == 0, :] = 1
    M[:, y_lens == 0] = 1

    x_lens[x_lens == 0] = 1
    y_lens[y_lens == 0] = 1
    M /= y_lens
    M = np.transpose(M)
    M /= x_lens
    M = np.transpose(M)
    return M