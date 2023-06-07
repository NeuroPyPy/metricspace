import numpy as np
from .calculate_spkd import spkd_functions
import importlib
rs_distances_spec = importlib.util.find_spec('rs_distances')

if rs_distances_spec is not None:
    try:
        import rs_distances
    except ImportError:
        print('Error during import of rs_distances. Fallback to spkd_v function.')
        rs_distances = None
else:
    rs_distances = None

def spkd(cspks: np.ndarray | list, qvals: list | np.ndarray, use_rs: bool = True):
    """
    Compute pairwise spike train distances with variable time precision for multiple cost values. 

    You can opt out of rust implementaion for easier debugging using the use_rs flag.

    Args:
        cspks (nested iterable[list | np.ndarray]): Each inner list contains spike times for a single spike train.
        qvals (list of float | int): List of time precision values to use in the computation.
        use_rs (bool, optional): Whether to use the rs_distances implementation. Defaults to True.

    Returns:
        ndarray: A 3D array containing pairwise spike train distances for each time precision value.
    """
    if rs_distances is not None and use_rs:
        d = rs_distances.calculate_spkd(cspks, qvals)
        return np.maximum(d, np.transpose(d, [1, 0, 2]))
    else:
        return spkd_functions.calculate_spkd(cspks, qvals, None)


def spkd_slide(cspks: np.ndarray | list, qvals: list | np.ndarray, res: float | int = 1e-3):
    """
    Compute pairwise spike train distances with variable time precision for multiple cost values,
    incorporating sliding of one spike train along the time axis.

    Currently only the python implementation is supported.
    TODO: Add rust implementation.

    Args:
        cspks (nested iterable[list | np.ndarray]): Each inner list contains spike times for a single spike train.
        qvals (list of float | int): List of time precision values to use in the computation.
        res (float, optional): The resolution of the sliding operation. Default is 1e-3.

    Returns:
        ndarray: A ND array containing pairwise spike train distances where N=len(costs), for each time precision value.
    """
    return spkd_functions.calculate_spkd(cspks, qvals, res)
