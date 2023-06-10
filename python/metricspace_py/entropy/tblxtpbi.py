import numpy as np
from . import histtpbi as htpbi


def tblxtpbi(ctabl, useall=0):
    """
        Calculates the Treves-Panzeri bias correction to be added to the naive transinformation estimate, in bits.

        Args:
            ctabl (numpy.ndarray): 2-dimensional array of counts.
            useall (int, optional): Use all bins or specific bins based on the parameter. Default is 0.

        Returns:
            float: Treves-Panzeri bias correction estimate in bits.
        """
    h = None
    if useall == 1:
        h = htpbi.histtpbi(np.sum(ctabl, axis=0), 1) + htpbi.histtpbi(np.sum(ctabl, axis=1), 1) - htpbi.histtpbi(ctabl,
                                                                                                                 1)
    elif useall == 0:
        pruned = ctabl[np.where(np.sum(ctabl, axis=1) > 0), :][:, np.where(np.sum(ctabl, axis=0) > 0)]
        h = htpbi.histtpbi(np.sum(pruned, axis=0), 1) + htpbi.histtpbi(np.sum(pruned, axis=1), 1) - htpbi.histtpbi(
            pruned, 1)
    elif useall == -1:
        h = htpbi.histtpbi(np.sum(ctabl, axis=0), 0) + htpbi.histtpbi(np.sum(ctabl, axis=1), 0) - htpbi.histtpbi(ctabl,
                                                                                                                 0)
    return h
