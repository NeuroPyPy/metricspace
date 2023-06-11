import numpy as np
import warnings
from .calculate_spkd.spkd_functions import calculate_spkd_py
from metricspace.metricspace_rs import calculate_spkd_rs
import pandas as pd


def spkd(cspks: list, qvals: list | np.ndarray, use_rs: bool = True):
    """
    Compute pairwise spike train distances with variable time precision for multiple cost values.

    Parameters
    ----------
    cspks : list[np.ndarray or list] 
        List where each inner iterable contains spike times (floats or ints) for a single spike train.
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

    Raises
    ------
    ValueError
        If cspks is not a list or if it contains less than 2 spike trains.

    Notes
    -----
    The Rust implementation speed improvement typically scales by the number of spike-trains in cspks.
    You can opt out for easier debugging using the use_rs flag. 

    """
    if not isinstance(cspks, list):
        raise ValueError("cspks must be a list.")
    if len(cspks) < 2:
        raise ValueError("cspks must contain at least 2 spike trains for comparisons.")
    if not isinstance(qvals, np.ndarray):
        qvals = np.array(qvals)
    if use_rs:
        d = calculate_spkd_rs(cspks, qvals)
        return np.maximum(d, np.transpose(d, [1, 0, 2]))
    else:
        return calculate_spkd_py(cspks, qvals, None, False)


def spkd_slide(
    cspks: list, qvals: list | np.ndarray, res: float | int = 1e-3, return_min_res: bool = False,
):
    """

    Compute pairwise spike train distances with variable time precision for multiple cost values.

    This is a modification of the original `cost-based spike-distance metric <http://www-users.med.cornell.edu/~jdvicto/metricdf.html#introduction>`
    that returns the minimum distance between two spike-trains over multiple possible time-translations of one of the spike-trains.
    
    This is helpful when you want to align spikes-trains with different window sizes, i.e. spike-train A is 2s, spike-train B is 1s, 
    and you want to find the best alignment between the two. 

    This method has the exact same "interpretation" of the original metric, but in a more comprehensive and laborious way.

    Parameters
    ----------
    cspks : list[np.ndarray or list] 
        List where each inner iterable contains spike times (floats or ints) for a single spike train.
    qvals : list or np.ndarray
        List or array of time precision values (floats or ints) to use in the computation.
    res : float or int, optional
        Time resolution (float or int) to use in the computation. Defaults to 1e-3, which indicates
        a millisecond resolution search window.
    return_min_res : bool, optional
        Whether to return the minimum distance and time translation for each spike-train pair. Defaults to False.


    Returns
    -------
    ndarray
        A 3D ndarray with floats representing pairwise spike train distances
        for each time precision value.
    tuple, optional
        A tuple containing the minimum distance and time translation for each spike-train pair. Only returned if return_min_res is True.

    Notes
    -----
    Currently, this function only uses the Python implementation.

    Raises
    ------
    ValueError
        If cspks is not a list or if it contains less than 2 spike trains.

    UserWarning
       If the resolution is too small, the computation time may be long. A resolution of 1e-4 adds 200 computations per spike pair.

    """
    if not isinstance(cspks, list):
        raise ValueError("cspks must be a list.")
    if len(cspks) < 2:
        raise ValueError("cspks must contain at least 2 spike trains for comparisons.")
    if not isinstance(qvals, np.ndarray):
        qvals = np.array(qvals)
    if res < 1e-4:
        raise UserWarning(f"Too small of a search window can drastically increase computation time: {res}")
    return calculate_spkd_py(cspks, qvals, res, return_min_res)
