import numpy as np
from numba import jit
import rs_distances as rsd


@jit(nopython=True, fastmath=True)
def iterate_spiketrains(scr, sd):
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


def spkd_v_rs(scr, sd):
    scr = rsd.iterate_spiketrains_impl(scr, sd)
    # The last column represents the final values of the accumulated cost of aligning the two spike trains
    d = np.squeeze(scr[:, -1, -1]).astype('float32')
    return d


def spkd_v(scr, sd):
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
    scr = iterate_spiketrains(scr, sd)

    # The last column represents the final values of the accumulated cost of aligning the two spike trains
    d = np.squeeze(scr[:, -1, -1]).astype('float32')

    return d


def spkd_pw_slide(cspks: np.ndarray | list, qvals: list | np.ndarray, res: float | int = 1e-3):
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
            # Check if both spike trains have at least one spike

            # TODO: Distinguish this inner loop, the same computation as in spkd
            if curcounts[xi] != 0 and curcounts[xj] != 0:
                # Use np.array here, to ensure a copy is made
                spk_train_a = np.array(cspks[xi])
                spk_train_b = np.array(cspks[xj])

                # Compute pairwise distances for each time offset in the range [-1, 1] with step size res
                for offset in np.arange(-1, 1 + res, res):
                    # Offset the spike times of the second spike train by the spike-time resolution given (res)
                    spk_train_a = spk_train_a + offset

                    # Compute absolute differences between all pairs of spike times in spk_train_a and spk_train_b
                    outer_diff = np.abs(spk_train_a.reshape(-1, 1) - spk_train_b.reshape(1, -1))

                    # Compute scaled difference by multiplying with qvals
                    sd = qvals.reshape((-1, 1, 1)) * outer_diff

                    # Initialize 3D array to hold intermediate computations
                    scr = np.zeros((len(qvals), curcounts[xi] + 1, curcounts[xj] + 1))

                    # Update scr along its second and third dimensions
                    scr[:, 1:, 0] += np.arange(1, curcounts[xi] + 1)
                    scr[:, 0, 1:] += np.arange(1, curcounts[xj] + 1).reshape((1, -1))

                    # Compute pairwise distance for current pair of spike trains and store in d
                    d_current = spkd_v(scr, sd)
                    d[xi, xj, :] = np.minimum(d[xi, xj, :], d_current) if offset != -1 else d_current
            else:
                # If either of the spike trains is empty, store the maximum spike count in d
                d[xi, xj, :] = max(curcounts[xi], curcounts[xj])

    # Ensure symmetry of pairwise distances by taking the maximum of d and its transpose along the first two dimensions
    return np.maximum(d, np.transpose(d, [1, 0, 2]))


def spkd_pw(cspks: np.ndarray | list, qvals: list | np.ndarray, res: float = 1e-4):
    """
       Compute pairwise spike train distances with variable time precision for multiple cost values.

         This function computes the pairwise distances between spike trains, considering
         different time precision levels specified by `qvals`. Differes from spkd in that it vectorizes the
         computation of the dynamic programming matrix for each q value intrinsically,
         and incorperates spike train iteration. Numba is used to drastically speed up the computation.

        see also: spkd, spkd_v, iterate_spiketrains

       Args:
           cspks (nested iterable[list | np.ndarray]): Each inner list contains spike times for a single spike train.
           qvals (list of float | int): List of time precision values to use in the computation.
           res (float, optional): The search resolution of the spike trains. Defaults to 1e-4.

       Returns:
           ndarray: A 3D array containing pairwise spike train distances for each time precision value.
       """
    if not hasattr(cspks, '__iter__') and not hasattr(cspks, '__len__'):
        # Check if cspks is an iterable object and countable
        raise TypeError('cspks must be an iterable, countable object')
    if not hasattr(qvals, '__iter__') and not hasattr(qvals, '__len__'):
        # Check if qvals is an iterable object and countable
        raise TypeError('qvals must be an iterable, countable object')
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
            # Check if both spike trains have at least one spike
            if curcounts[xi] != 0 and curcounts[xj] != 0:
                # Convert spike times and qvals to numpy arrays for cleanliness
                spk_train_a = np.array(cspks[xi])
                spk_train_b = np.array(cspks[xj])
                # ----- Start: To Slide or Not to Slide -------

                # Compute absolute differences between all pairs of spike times in spk_train_a and spk_train_b
                outer_diff = np.abs(spk_train_a.reshape(-1, 1) - spk_train_b.reshape(1, -1))

                # Compute scaled difference by multiplying with qvals
                sd = qvals.reshape((-1, 1, 1)) * outer_diff

                # Initialize 3D array to hold intermediate computations
                scr = np.zeros((len(qvals), curcounts[xi] + 1, curcounts[xj] + 1))

                # Update scr along its second and third dimensions
                scr[:, 1:, 0] += np.arange(1, curcounts[xi] + 1)
                scr[:, 0, 1:] += np.arange(1, curcounts[xj] + 1).reshape((1, -1))

                # Compute pairwise distance for current pair of spike trains and store in d
                d[xi, xj, :] = spkd_v(scr, sd)
            else:
                # If either of the spike trains is empty, store the maximum spike count in d
                d[xi, xj, :] = max(curcounts[xi], curcounts[xj])

    # Ensure symmetry of pairwise distances by taking the maximum of d and its transpose along the first two dimensions
    return np.maximum(d, np.transpose(d, [1, 0, 2]))


def spkd(tli: np.ndarray | list, tlj: np.ndarray | list, cost: float | int):
    """
    Compute the spike-time distance between two spike trains for a given cost (q).

    This function computes the spike-time distance between two spike trains,
    based on a cost function parameter that influences the penalty for differences
    in spike times. The cost parameter can take values from 0 to infinity, where
    a cost of 0 gives the absolute difference in the number of spikes and a cost of
    infinity gives the total number of spikes.

    Args:
        tli (list of int): The spike times for the first spike train.
        tlj (list of int): The spike times for the second spike train.
        cost (float): A parameter for the cost function determining the penalty for
                      differences in spike times.

    Returns:
        float: The computed spike-time distance between the two spike trains.
    """
    if not hasattr(tli, '__iter__') and not hasattr(tli, '__len__'):
        # Check if tli is an iterable object and countable
        raise TypeError('tli must be an iterable, countable object')
    if not hasattr(tlj, '__iter__') and not hasattr(tlj, '__len__'):
        # Check if tlj is an iterable object and countable
        raise TypeError('tlj must be an iterable, countable object')
    if not isinstance(cost, (float, int)):
        # Check if cost is a float or int
        raise TypeError('cost must be a float or int')

    # Number of spikes in each spike train
    nspi = len(tli)
    nspj = len(tlj)

    # Check for edge cases where the cost is either 0 or infinity
    if cost == 0:
        return abs(nspi - nspj)
    elif cost == np.inf:
        return nspi + nspj

    # Initialize the scoring matrix with dimension (nspi + 1) x (nspj + 1)
    scr = np.zeros((nspi + 1, nspj + 1))
    scr[:, 0] = np.arange(nspi + 1)
    scr[0, :] = np.arange(nspj + 1)

    # If both spike trains have at least one spike
    if nspi and nspj:
        # Loop over the scoring matrix and update its values based on the minimum
        # between the current value and the values in the preceding row and column
        for i in range(1, nspi + 1):
            for j in range(1, nspj + 1):
                scr[i, j] = min(scr[i - 1, j] + 1, scr[i, j - 1] + 1,
                                scr[i - 1, j - 1] + cost * np.abs(tli[i - 1] - tlj[j - 1]))

    # Return the final value in the scoring matrix, which represents the spike-time distance
    return scr[-1, -1]


def spkd_slide(tli: np.ndarray | list, tlj: np.ndarray | list, cost: float | int, res: float | int = 1e-3):
    """
    Compute the optimal spike-time distance by sliding one spike train along the time axis.

    This function computes the minimum spike-time distance between two spike trains by sliding one spike train
    along the time axis and evaluating the spike-time distance at each step. The cost of shifting the spike times
    is given by the parameter `cost`. The resolution of the sliding operation can be specified by `res`.

    Args:
        tli (list of int): The spike times for the first spike train.
        tlj (list of int): The spike times for the second spike train.
        cost (float): A parameter for the cost function determining the penalty for differences in spike times.
        res (float, optional): The resolution of the sliding operation. Default is 1e-3.

    Returns:
        tuple: A tuple containing the minimum spike-time distance, the amount of slide at which this minimum is achieved,
               a list of the spike-time distances for each slide, and a list of the slide values.
    """
    if not hasattr(tli, '__iter__') and not hasattr(tli, '__len__'):
        # Check if tli is an iterable object and countable
        raise TypeError('tli must be an iterable, countable object')
    if not hasattr(tlj, '__iter__') and not hasattr(tlj, '__len__'):
        # Check if tlj is an iterable object and countable
        raise TypeError('tlj must be an iterable, countable object')
    if not isinstance(cost, (float, int)):
        # Check if cost is a float or int
        raise TypeError('cost must be a float or int')
    if not isinstance(res, (float, int)):
        # Check if res is a float or int
        raise TypeError('res must be a float or int')

    # If one of the spike trains is empty, return the total number of spikes
    if not tli or not tlj:
        return len(tli) + len(tlj), [], [], []

    # Determine the range of sliding values based on the maximum and minimum spike times
    s_list = np.arange(((-max(tli) + min(tlj) - res) // res) * res,
                       ((max(tlj) - min(tli) + res) // res) * res, res)

    # Compute the spike-time distance for each sliding value
    d_list = [spkd(tli + s, tlj, cost) for s in s_list]

    # Determine the minimum spike-time distance and the corresponding slide value
    d = min(d_list)
    s_min = s_list[np.argmin(d_list)]

    return d, s_min, d_list, s_list
