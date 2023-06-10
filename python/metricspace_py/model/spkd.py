import numpy as np
from .calculate_spkd.spkd_functions import calculate_spkd_py
from ..metricspace_rs import calculate_spkd_rust
import pandas as pd


def spkd(cspks: np.ndarray | list, qvals: list | np.ndarray, use_rs: bool = True):
    """
    Compute pairwise spike train distances with variable time precision for multiple cost values.

    Parameters
    ----------
    cspks : list or np.ndarray
        Nested iterable. Each inner list or np.ndarray contains spike times
        (floats or ints) for a single spike train.
    qvals : list or np.ndarray
        List or array of time precision values (floats or ints) to use in the computation.
    use_rs : bool, optional
        Whether to use the rs_distances implementation. If True, it utilizes
        calculate_spkd_rust function, otherwise, it uses spkd_functions.calculate_spkd.
        Defaults to True.
  
    Returns
    -------
    ndarray
        A 3D ndarray with floats representing pairwise spike train distances
        for each time precision value.

    Notes
    -----
    The Rust implementation is typically faster, but you can opt out for easier
    debugging using the use_rs flag. Note that the behavior and performance
    may differ based on this flag.
    """
    if use_rs:
        d = calculate_spkd_rust(cspks, qvals)
        return np.maximum(d, np.transpose(d, [1, 0, 2]))
    else:
        return calculate_spkd_py(cspks, qvals, None)


def spkd_slide(
    cspks: np.ndarray | list, qvals: list | np.ndarray, res: float | int = 1e-3
):
    """
    Compute pairwise spike train distances with variable time precision for multiple cost values.

    This is a modification of the original `cost-based spike-distance metric <http://www-users.med.cornell.edu/~jdvicto/metricdf.html#introduction>`
    that returns the minimum distance between two spike-trains over multiple possible time-translations of one of the spike-trains.
    
    This is helpful when you want to align spikes-trains with different window sizes, i.e. spike-train A is 2s, spike-train B is 1s, 
    and you want to find the best alignment between the two.


    Parameters
    ----------
    cspks : list or np.ndarray
        Nested iterable. Each inner list or np.ndarray contains spike times
        (floats or ints) for a single spike train.
    qvals : list or np.ndarray
        List or array of time precision values (floats or ints) to use in the computation.
    res : float or int, optional
        Time resolution (float or int) to use in the computation. Defaults to 1e-3, which indicates
        a millisecond resolution search window.

    Returns
    -------
    ndarray
        A 3D ndarray with floats representing pairwise spike train distances
        for each time precision value.

    Notes
    -----
    Currently, this function only uses the Python implementation.

    """
    return _spkd_functions._calculate_spkd(cspks, qvals, res)
