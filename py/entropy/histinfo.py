import numpy as np


def histinfo(pvec):
    pnz = pvec.flatten()
    pnz = pnz[pnz > 0]

    if len(pnz) == 0:
        h = 0
    else:
        h = -np.dot(pnz, np.log2(pnz))

    return h
