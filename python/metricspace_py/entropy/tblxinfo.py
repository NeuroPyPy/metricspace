import numpy as np
from . import histinfo as hinfo


def tblxinfo(tabl):
    """
    Calculates the transinformation, in bits.

    Args:
        tabl (numpy.ndarray): 2-dimensional array of probabilities.

    Returns:
        float: Transinformation in bits.
    """
    h = hinfo.histinfo(np.sum(tabl, axis=1)) + hinfo.histinfo(np.sum(tabl, axis=0)) - hinfo.histinfo(tabl)
    return h
