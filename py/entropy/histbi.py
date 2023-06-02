from . import histjabi
from . import histtpbi
from . import histcsbi

def histbi(cvec, type, param=None):
    if type == 'ja':
        h = histjabi(cvec)
    elif type == 'tp':
        useall = 0 if param is None else param
        h = histtpbi(cvec, useall)
    elif type == 'cs':
        h = histcsbi(cvec, 'least')
    return h