import math
import numpy as np
from scipy.linalg import hadamard as scipy_hadamard


def polygonal2d(num_classes=10):
    vec = np.zeros((2, num_classes))
    for n in range(0, num_classes):
        vec[0, n] = 1 * math.cos(2 * math.pi * n / num_classes)
        vec[1, n] = 1 * math.sin(2 * math.pi * n / num_classes)
    vec = vec.transpose()
    return vec


def hadamard(num_classes=10, feat_dim=10):
    sz = 2 ** int(math.ceil(math.log(max(feat_dim, num_classes), 2)))

    # Constructs an n-by-n Hadamard matrix, using Sylvester's construction.
    # n must be a power of 2.
    vec = scipy_hadamard(sz)
    vec = vec[:num_classes, :feat_dim]
    return vec


def dcube(num_classes=16, feat_dim=4):

    target_dim = 2 ** feat_dim
    if num_classes != target_dim:
        raise ValueError("num_classes not a power of 2.. changing to: {}".format(target_dim))
    if feat_dim != int(np.ceil(np.log2(num_classes)).astype(np.int32)):
        raise ValueError("wrong feat_dim")

    # Build the vertices of the hypercube
    # 2**feat_dim directions
    def perm(n):
        return np.array([np.array([(1, -1)[(t >> i) % 2] for i in range(n)]) for t in range(2 ** n)])
    vec = perm(feat_dim)
    # normalize the vector!
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    return vec


def orthoplex(num_classes=10, feat_dim=5):

    if feat_dim != np.ceil(num_classes / 2).astype(int):
        raise ValueError("wrong number of feat_dim")

    vec = np.identity(feat_dim)
    vec = np.vstack((vec, -vec))
    return vec


def dsimplex(num_classes=10):
    def simplex_coordinates2(m):
        # add the credit
        import numpy as np

        x = np.zeros([m, m + 1])
        for j in range(0, m):
            x[j, j] = 1.0

        a = (1.0 - np.sqrt(float(1 + m))) / float(m)

        for i in range(0, m):
            x[i, m] = a

        #  Adjust coordinates so the centroid is at zero.
        c = np.zeros(m)
        for i in range(0, m):
            s = 0.0
            for j in range(0, m + 1):
                s = s + x[i, j]
            c[i] = s / float(m + 1)

        for j in range(0, m + 1):
            for i in range(0, m):
                x[i, j] = x[i, j] - c[i]

        #  Scale so each column has norm 1. UNIT NORMALIZED
        s = 0.0
        for i in range(0, m):
            s = s + x[i, 0] ** 2
        s = np.sqrt(s)

        for j in range(0, m + 1):
            for i in range(0, m):
                x[i, j] = x[i, j] / s

        return x

    feat_dim = num_classes - 1
    ds = simplex_coordinates2(feat_dim)
    return ds.transpose()
