from . import histjabi as hjabi
from . import histtpbi as htpbi


def histbi(cvec, estimate_type, param=None):
    """
      Calculates the bias estimate for the naive plugin histogram entropy estimate, in bits.

      Args:
          cvec (numpy.ndarray): Vector of counts.
          estimate_type (str): Type of bias estimation. 'ja' for jackknife estimate, 'tp' for Treves-Panzeri estimate.
            - 'ja' uses the jackknife estimate of the bias, which is the more conservative option.
            
          param (int, optional): Additional parameter for specific bias estimation. Default is None.

      Returns:
          float: Bias estimate in bits.
      """
    h = None
    if estimate_type == 'ja':
        h = hjabi.histjabi(cvec)
    elif type == 'tp':
        useall = 0 if param is None else param
        h = htpbi.histtpbi(cvec, useall)
    return h
