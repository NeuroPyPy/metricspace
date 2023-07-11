import numpy as np
from . import histinfo


def tblxinfo(tabl):
    """
    Calculates the transinformation, in bits.

    Args:
        tabl (numpy.ndarray): 2-dimensional array of probabilities.

    Returns:
        float: Transinformation in bits.
    """
    h = histinfo(np.sum(tabl, axis=1)) + histinfo(np.sum(tabl, axis=0)) - histinfo(tabl)
    return h
