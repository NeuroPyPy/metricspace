"""metricspace.distance.calculate_spkd.calculate_spkd.py

Contains internal spike distance calculation functions. These functions are
called by the public functions in metricspace.distance.spkd.py and metricspace.distance.spkd_slide.py.

They are private to clean up the public namespace, avoid circular imports, and prevent confusion about which
spike distance function is being called.

"""
import numpy as np
from numba import jit


# Outer Entrypoint for Python Implementation -----------------------------------------------------------------------------------------------------
def calculate_spkd_py(
    cspks: list, qvals: list | np.ndarray, res: float | int | None = 1e-2
):
    """
    Internal function to compute pairwise spike train distances with variable time precision for multiple cost values.

    Parameters
    ----------
    cspks : list[np.ndarray or list] 
        List where each inner iterable contains spike times (floats or ints) for a single spike train.
    qvals : list or np.ndarray
        List or array of time precision values to use in the computation.
    res : float, optional
        The search resolution of the spike trains. Defaults to 1e-4.

    Returns
    -------
    ndarray
        A 3D array containing pairwise spike train distances for each time precision value.

    """
    # Calculate the count of spikes in each spike train
    curcounts = [len(x) for x in cspks]
    numt = len(cspks)

    # Initialize 3D array to store pairwise distances for each time precision
    d = np.zeros((numt, numt, len(qvals)))
    offsets = np.arange(-1, 1 + res, res) if res else [0]

    # Iterate over all pairs of spike trains
    for xi in range(numt - 1):
        for xj in range(xi + 1, numt):
            if curcounts[xi] != 0 and curcounts[xj] != 0:
                spk_train_a = np.array(cspks[xi])
                spk_train_b = np.array(cspks[xj])

                first_iter = True
                d_min = np.inf  # protect against unbound local error

                for offset in offsets:
                    spk_train_a_copy = spk_train_a.copy()
                    if res:
                        spk_train_a_copy += offset
                    outer_diff = np.abs(
                                        spk_train_a_copy.reshape(-1, 1) - spk_train_b.reshape(1, -1)
                                    )
                    sd = qvals.reshape((-1, 1, 1)) * outer_diff
                    scr = np.zeros((len(qvals), curcounts[xi] + 1, curcounts[xj] + 1))
                    scr[:, 1:, 0] += np.arange(1, curcounts[xi] + 1)
                    scr[:, 0, 1:] += np.arange(1, curcounts[xj] + 1).reshape((1, -1))

                    d_current = _compute_spiketrain_distance_py(scr, sd)

                    # keep a running minimum, likely a better way to do this, but this seems the most clear
                    if first_iter:
                        d_min = d_current
                        first_iter = False
                    else:
                        d_min = np.minimum(d_min, d_current)
                d[xi, xj, :] = d_min
            else:
                d[xi, xj, :] = max(curcounts[xi], curcounts[xj])
    return np.maximum(d, np.transpose(d, [1, 0, 2]))


def _compute_spiketrain_distance_py(scr, sd):
    """
    Compute spike-time distance.

    This function calculates the spike-time distance using the `iter_scr_numba` function to update the `scr` array
    and then return the final values in the last column of the last 2D slice of the `scr` array.

    Args:
        scr (numpy.ndarray): A 3D array that gets updated in the process. Each 2D slice of the array corresponds
                             to a different cost factor, and the elements within each slice represent the accumulated
                             cost of aligning the two spike trains up to that point.
        sd (numpy.ndarray): A 3D array used in the computation of the quantities. Each 2D slice of this array represents
                            the cost of aligning each pair of spikes from the two spike trains for a different cost factor.

    Returns:
        numpy.ndarray: A 1D array representing the spike-time distances.
    """
    # Need to separate this iteration for compatibility with numba
    scr = _distance_optimized_py(scr, sd)

    # The last column represents the final values of the accumulated cost of aligning the two spike trains
    d = np.squeeze(scr[:, -1, -1]).astype("float32")

    return d


@jit(nopython=True, fastmath=True)
def _distance_optimized_py(scr, sd):
    """
    Perform an iteration over 2D slices of the 3D scr and sd arrays.

    The scr array is a 3D array storing cost values at different steps of the computation, and the sd array
    is a 3D array that stores pairwise differences between two spike trains multiplied by a cost factor.

    This function iterates over the second and third dimensions of the 3D scr and sd arrays and updates each
    element in the scr array to the minimum value of three quantities computed from the scr and sd arrays.

    Args:
        scr (numpy.ndarray): 2D slice of the array corresponding to the accumulated cost of aligning two spike trains
                             to a different cost factor, and the elements within each slice represent the accumulated
                             cost of aligning the two spike trains up to that point.
        sd (numpy.ndarray): A 3D array used in the computation of the quantities.
                            Each 2D slice of this array represents sums of the cost of aligning each pair of spikes
                            from the two spike trains for a different cost factor.

    Returns:
        numpy.ndarray: The updated scr array.
    """
    # Iterating over the second and third dimensions of scr and sd
    for xii in range(1, sd.shape[1] + 1):
        for xjj in range(1, sd.shape[2] + 1):
            # Compute the three quantities
            a = scr[:, xii - 1, xjj] + 1
            b = scr[:, xii, xjj - 1] + 1
            c = scr[:, xii - 1, xjj - 1] + sd[:, xii - 1, xjj - 1]

            # Update the scr array with the minimum of the three quantities
            scr[:, xii, xjj] = np.minimum(a, np.minimum(b, c))

    return scr
