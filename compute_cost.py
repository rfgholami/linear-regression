import numpy as np


def compute_cost(X, y, theta):
    m = X.shape[0]
    h = np.dot(X, theta)
    cost = np.multiply((1. / (2. * m)), np.sum(np.power((h - y), 2), axis=0))
    return cost
