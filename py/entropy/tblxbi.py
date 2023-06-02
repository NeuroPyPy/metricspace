import numpy as np
from . import histbi as hbi
from . import tblxtpbi


def tblxbi(ctabl, type, param=None):
    ty = type[0:2].lower()
    if ty == 'ja':
        h = hbi.histbi(np.sum(ctabl, axis=0), 'ja') + hbi.histbi(np.sum(ctabl, axis=1), 'ja') - hbi.histbi(ctabl, 'ja')
    elif ty == 'tr':
        useall = 0 if param is None else param
        h = tblxtpbi(ctabl, useall)
    return h
