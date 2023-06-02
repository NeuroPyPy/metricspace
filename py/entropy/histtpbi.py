import numpy as np


def histtpbi(cvec, useall=0):
    counts = np.reshape(cvec, (1, np.prod(cvec.shape)))
    if np.sum(counts) == 0:
        h = 0
        return h
    if useall:
        bins = len(counts)
    else:
        bins = np.sum(counts > 0)
    h = (bins - 1) / (2 * np.sum(counts) * np.log(2))
    return h
