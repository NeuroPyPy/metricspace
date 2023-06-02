import numpy as np
from . import histbi as hbi
from . import tblxtpbi as ttpbi


def tblxbi(ctabl, estimate_type, param=None):
    """
        Calculates the Treves-Panzeri or jackknife bias correction to be added to the naive transinformation estimate, in bits.

        Args:
            ctabl (numpy.ndarray): 2-dimensional array of counts.
            estimate_type (str): Type of bias estimation. 'ja' for jackknife estimate, 'tr' for Treves-Panzeri estimate.
            param (int, optional): Additional parameter for specific bias estimation. Default is None.

        Returns:
            float: Bias correction estimate in bits.
        """
    h = None
    ty = estimate_type[0:2].lower()
    if ty == 'ja':
        h = hbi.histbi(np.sum(ctabl, axis=0), 'ja') + hbi.histbi(np.sum(ctabl, axis=1), 'ja') - hbi.histbi(ctabl, 'ja')
    elif ty == 'tr':
        useall = 0 if param is None else param
        h = ttpbi.tblxtpbi(ctabl, useall)
    return h
