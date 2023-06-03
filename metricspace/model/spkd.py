import numpy as np
from .calculate_spkd import spkd_functions


def spkd(cspks: np.ndarray | list, qvals: list | np.ndarray):
    """
    Compute pairwise spike train distances with variable time precision for multiple cost values.

    Args:
        cspks (nested iterable[list | np.ndarray]): Each inner list contains spike times for a single spike train.
        qvals (list of float | int): List of time precision values to use in the computation.

    Returns:
        ndarray: A 3D array containing pairwise spike train distances for each time precision value.
    """
    return spkd_functions.calculate_spkd(cspks, qvals, None)


def spkd_slide(cspks: np.ndarray | list, qvals: list | np.ndarray, res: float | int = 1e-3):
    """
    Compute pairwise spike train distances with variable time precision for multiple cost values,
    incorporating sliding of one spike train along the time axis.

    Args:
        cspks (nested iterable[list | np.ndarray]): Each inner list contains spike times for a single spike train.
        qvals (list of float | int): List of time precision values to use in the computation.
        res (float, optional): The resolution of the sliding operation. Default is 1e-3.

    Returns:
        ndarray: A ND array containing pairwise spike train distances where N=len(costs), for each time precision value.
    """
    return spkd_functions.calculate_spkd(cspks, qvals, res)
