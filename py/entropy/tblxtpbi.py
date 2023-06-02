import numpy as np
from . import histtpbi as htpbi


def tblxtpbi(ctabl, useall=0):
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
