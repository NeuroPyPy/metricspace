import numpy as np


def histtpbi(cvec, useall=0):
    """
        Calculates the Treves-Panzeri bias that should be added to the naive plugin estimate, in bits.

        Args:
            cvec (numpy.ndarray): Vector of counts.
            useall (int, optional): Use all bins or specific bins based on the parameter. Default is 0.

        Returns:
            float: Treves-Panzeri bias in bits.
        """
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
