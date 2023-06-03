"""metricspace.distance.calculate_spkd.calculate_spkd.py

Contains internal spike distance calculation functions. These functions are
called by the public functions in metricspace.distance.spkd.py and metricspace.distance.spkd_slide.py.

They are private to clean up the public namespace, avoid circular imports, and prevent confusion about which
spike distance function is being called.

"""
import numpy as np
import importlib.util
from numba import jit

rs_distances_spec = importlib.util.find_spec('rs_distances')

if rs_distances_spec is not None:
    try:
        import rs_distances
    except ImportError:
        print('Error during import of rs_distances. Fallback to spkd_v function.')
        rs_distances = None
else:
    rs_distances = None


# Outer Entrypoint -----------------------------------------------------------------------------------------------------
def calculate_spkd(cspks: np.ndarray | list, qvals: list | np.ndarray, res: float | int | None = 1e-4):
    """
       Internal function to compute pairwise spike train distances with variable time precision for multiple cost values.

       Args:
           cspks (nested iterable[list | np.ndarray]): Each inner list contains spike times for a single spike train.
           qvals (list of float | int): List of time precision values to use in the computation.
           res (float, optional): The search resolution of the spike trains. Defaults to 1e-4.

       Returns:
           ndarray: A 3D array containing pairwise spike train distances for each time precision value.
       """
    if not isinstance(qvals, np.ndarray):
        # Check if qvals is a numpy array
        qvals = np.array(qvals)

    # Calculate the count of spikes in each spike train
    curcounts = [len(x) for x in cspks]
    numt = len(cspks)

    # Initialize 3D array to store pairwise distances for each time precision
    d = np.zeros((numt, numt, len(qvals)))

    # Iterate over all pairs of spike trains
    for xi in range(numt - 1):
        for xj in range(xi + 1, numt):
            if curcounts[xi] != 0 and curcounts[xj] != 0:
                spk_train_a = np.array(cspks[xi])
                spk_train_b = np.array(cspks[xj])

                offsets = np.arange(-1, 1 + res, res) if res else [0]
                for offset in offsets:
                    if res:
                        spk_train_a = spk_train_a + offset

                    outer_diff = np.abs(spk_train_a.reshape(-1, 1) - spk_train_b.reshape(1, -1))
                    sd = qvals.reshape((-1, 1, 1)) * outer_diff
                    scr = np.zeros((len(qvals), curcounts[xi] + 1, curcounts[xj] + 1))
                    scr[:, 1:, 0] += np.arange(1, curcounts[xi] + 1)
                    scr[:, 0, 1:] += np.arange(1, curcounts[xj] + 1).reshape((1, -1))

                    d_current = _compute_spike_distance(scr, sd)
                    d[xi, xj, :] = np.minimum(d[xi, xj, :], d_current) if offset != -1 else d_current
            else:
                d[xi, xj, :] = max(curcounts[xi], curcounts[xj])
    return np.maximum(d, np.transpose(d, [1, 0, 2]))


# Check if rs is installed ---------------------------------------------------------------------------------------------
def _compute_spike_distance(scr, sd):
    """
    Compute spike-time distance, using either the rs_distances module (if installed and importable) or a fallback function.

    Args:
        scr, sd: Input arguments for the distance computation function.

    Returns:
        numpy.ndarray: The result of the spike-time distance computation.
    """
    if rs_distances is not None:
        return _spkd_v_rs(scr, sd)
    else:
        return _spkd_v(scr, sd)


# If rs is installed ---------------------------------------------------------------------------------------------------
def _spkd_v_rs(scr, sd):
    scr = rs_distances.iterate_spiketrains_impl(scr, sd)
    # The last column represents the final values of the accumulated cost of aligning the two spike trains
    d = np.squeeze(scr[:, -1, -1]).astype('float32')
    return d


# If rs not installed --------------------------------------------------------------------------------------------------
def _spkd_v(scr, sd):
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
    scr = _iterate_spiketrains(scr, sd)

    # The last column represents the final values of the accumulated cost of aligning the two spike trains
    d = np.squeeze(scr[:, -1, -1]).astype('float32')

    return d


@jit(nopython=True, fastmath=True)
def _iterate_spiketrains(scr, sd):
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
