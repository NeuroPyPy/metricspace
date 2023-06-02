import numpy as np


def histinfo(pvec):
    """
       Calculates the histogram information, in bits.

       Args:
           pvec (numpy.ndarray): Vector of probabilities.

       Returns:
           float: Histogram information in bits.
       """
    pnz = pvec.flatten()
    pnz = pnz[pnz > 0]

    if len(pnz) == 0:
        h = 0
    else:
        h = -np.dot(pnz, np.log2(pnz))

    return h
