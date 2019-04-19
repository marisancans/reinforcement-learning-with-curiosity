from numpy import dot
import numpy as np
from numpy.linalg import norm

import sklearn.metrics.pairwise

# normalized 0 .. 2 (0 ~ same)
def cosine_similarity_(a, b):
    if np.sum(a) == 0 or np.sum(b) == 0:
        return 2.0
    return 1.0 - dot(a, b)/(norm(a)*norm(b))


# normalized 0 .. 2 (0 ~ same)
def cosine_similarity(a, b, reduce=np.average):
    if np.sum(a) == 0 or np.sum(b) == 0:
        return 2.0
    if len(a.shape) == 1:
        a = np.expand_dims(a, axis=0)
    if len(b.shape) == 1:
        b = np.expand_dims(b, axis=0)
    result = 1.0 - sklearn.metrics.pairwise.cosine_similarity(a, b)
    if result.shape[0] == 0:
        return float(result[0])
    return float(reduce(result))


def cross_product(a, b):
    return normalize_vec(np.cross(a, b))


def normalize_vec(v):
    sum = np.sum(v**2)
    if sum == 0:
        return v
    v = v / np.sqrt(sum)
    return v