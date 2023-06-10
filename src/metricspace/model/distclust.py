"""
Python translation of original distclust_fast.m

Translation by Flynn O'Connell, 2023.
(c) 2000 by Daniel Reich. All rights reserved.
"""
import numpy as np


def _validate_distclust_input(dists, nsam, ifresamp, iftrump):
    """ Validate input for distclust function. """
    assert (
        iftrump != 1 or ifresamp != 2
    ), "Cannot bootstrap resample when 0-distances trump."
    assert (
        dists.shape[0] == dists.shape[1]
    ), f"Input matrix should be square, given matrix has dimensions ({dists.shape[0]}, {dists.shape[1]})."
    assert np.all(np.diag(dists) == 0), "Input matrix should have 0s on the diagonal."
    assert np.all(
        nsam > 0
    ), "nsam should be a list or 1D array of positive integers representing the number samples for each class."
    assert (
        np.sum(nsam) == dists.shape[0] == dists.shape[1]
    ), f"Sum of nsam ({np.sum(nsam)}) should be the same as input matrix dimensions ({dists.shape[0]})."


def distclust(dists, nsam, expo=-2, ifresamp=0, iftrump=1):
    """
    Distance clustering. Classificatiion algorithm that returns a `KxK` confusion matrix where `K`
            is the number of classes.

    Parameters
    ----------
    dists : numpy.ndarray
        A 2D symmetric matrix of pairwise distances with 0's on the diagonal. Its size should be ``sum(nsam) x sum(nsam)``.
        This will be the output of the spkd function.
    nsam : list or numpy.ndarray
        A list or 1D array where each entry is the number of trials in each condition.
    expo : float or str, optional
        The exponent value for calculating the weighted mean distance. If expo is 0, it is set to 'median'. If expo is a
        negative number, iftrump is set to 1. If expo is 'median', the median distance is used instead of the weighted mean.
        Default is -2 (gravitationally-weighted mean).
    ifresamp : int, optional
        The resampling method to use. Options are 0 - No resampling, 1 - Relabel resampling, 2 - Bootstrap resampling.
    iftrump : int, optional
        Determines whether 0-distances should trump other values. Options are 0 - 0-distances do not trump other values,
        1 - 0-distances trump other values.

    Returns
    -------
    numpy.ndarray
        A confusion matrix of size `len(nsam)` x `len(nsam)`.

    Raises
    ------
    ValueError : If any of the following conditions are met:
        -  `iftrump` is 1 and `ifresamp` is 2, bootstrapping is undefined behavior.
        -  `dists` does not have 0's on the diagonal.
        -  `dists` is not a square matrix.
        -  `expo` is not a string or float.
        -  `nsam` is not a list or 1D array of positive integers.
        -  The sum of `nsam` is not equal to the number of rows and columns in `dists`.

    See Also
    --------
    metricspace.utils.spkd : Calculate pairwise spike train distances with variable time precision for multiple cost values.
                    This will be the input to distclust.
    numpy.unique : Find the unique elements of an array.
                   Easiest method to determine nsam. See Examples for more details.


    Notes
    -----
    The confusion matrix is calculated by averaging the pairwise distances between trials of the same class and
    comparing them to the pairwise distances between trials of different classes. The resulting matrix is
    y_pred vs y_true. Each row should sum to the number of trials/samples in each class.

    Algorithm doesn't assume all classes have the same number of trials and self-distances are not included in the calculation.

    Examples
    --------
        The easiest way to get nsam is to use np.unique(y, return_counts=True)[1], where y is a numpy array-like object (list, numpy.ndarray, etc.)
        that contains the class labels for each sample. For example, if y = [1, 1, 1, 2, 2, 3, 3, 3, 3], then nsam = [3, 2, 4].
        --------
        >>> import numpy as np
        >>> import metricspace as ms
        >>> distances = np.random.uniform(1, 12, size=(10, 10))
        >>> distances = np.array(np.triu(distances, k=1) + np.triu(distances, k=1).T)  # make it symmetric and zero diagonal
        >>> labels = ["A", "A", "A", "A", "B", "B", "C", "C", "C", "C"]
        >>> nsam = np.unique(labels, return_counts=True)[1]
        >>> print(nsam)
        [4 2 4]
        >>> confusion_matrix = ms.distclust(distances, nsam)
        >>> row_sums = np.sum(confusion_matrix, axis=1)
        >>> print(np.array_equal(row_sums, nsam))
        True
        >>> print(confusion_matrix.shape, len(nsam)) # these should all be the same!
        ((3, 3), 3)
    """
    if not isinstance(dists, np.ndarray):
        dists = np.array(dists)
    if not isinstance(nsam, np.ndarray):
        nsam = np.array(nsam)
    _validate_distclust_input(dists, nsam, ifresamp, iftrump)

    if expo == 0:
        expo = "median"
    if not isinstance(expo, str) and expo < 0:
        iftrump = 1

    ncla = len(nsam)  # number of Classes/Tastants
    anear = np.zeros((ncla, ncla))
    if ifresamp == 1:  # relabel resampling
        ind = np.random.permutation(dists.shape[1])
        dists = dists[ind, :][:, ind]
    elif ifresamp == 2:  # a bootstrap resampling
        ndx = []
        nsam2 = np.concatenate([[0], np.cumsum(nsam[:-1])])
        for isam in range(ncla):
            ndx.extend(
                nsam2[isam]
                + np.ceil(nsam[isam] * np.random.rand(nsam[isam])).astype(int)
            )
        dists = dists[ndx, :][:, ndx]
    samend = np.cumsum(nsam)
    samstart = np.concatenate([[0], samend[:-1]]) + 1

    av = np.zeros(
        (ncla, dists.shape[1])
    )  # average distance between trials of the same class
    if iftrump == 0:
        for i in range(ncla):
            if isinstance(expo, str):
                av[i, :] = np.median(dists[samstart[i] : samend[i], :], axis=0)
                # this ensures self-distances (diagonals) are not included in the calculation
                for j in range(samstart[i], samend[i]):
                    av[i, j] = np.median(
                        np.concatenate(
                            [dists[samstart[i] : j, j], dists[j + 1 : samend[i], j]]
                        )
                    )
            else:
                av[i, :] = (
                    np.sum(dists[samstart[i] : samend[i], :] ** expo) / nsam[i]
                ) ** (1 / expo)
                # this ensures self-distances (diagonals) are not included in the calculation
                for j in range(samstart[i], samend[i]):
                    # compute an average of values in subset of rows/columns excluding the diagonal
                    # and average distances from spike-train i to j, except if its negative
                    av[i, j] = (
                        np.sum(
                            np.concatenate(
                                [dists[samstart[i] : j, j], dists[j + 1 : samend[i], j]]
                            )
                            ** expo
                        )
                        / (nsam[i] - 1)
                    ) ** (1 / expo)
    else:
        dists[np.diag_indices_from(dists)] = np.nan
        for i in range(ncla):
            nsm = nsam[i] * np.ones(dists.shape[1])
            nsm[samstart[i] - 1 : samend[i]] = (
                nsam[i] - 1
            )  # -1 because python is 0-indexed
            temp = dists[
                samstart[i] - 1 : samend[i], :
            ]  # -1 because python is 0-indexed
            nz = temp < np.finfo(float).eps
            temp = np.zeros_like(temp)
            temp[nz] = 1
            av[i, :] = -np.sum(temp, axis=0) / nsm  # -fraction of zeros in each column
            nz = av[i, :] == 0
            nz_indices = np.where(nz)[
                0
            ]  # Get the indices where nz is True, columns with no zero values
            ident = np.where(
                (nz_indices >= samstart[i] - 1) & (nz_indices < samend[i])
            )[0]
            temp = dists[samstart[i] - 1 : samend[i], :][
                :, nz
            ]  # non-zero columns are not all ones
            if isinstance(expo, str):
                md = np.median(temp, axis=0)
                temp = np.sort(temp[:, ident])
                md[ident] = np.median(temp[: nsam[i] - 1, :])
            else:
                # weighted mean distance, train to other classes
                md = (np.sum(temp**expo, axis=0) / nsam[i]) ** (1 / expo)
                temp = np.sort(
                    temp[:, ident], axis=0
                )  # compared to matlabs boolean indexing syntax
                md[ident] = (
                    np.sum(temp[: nsam[i] - 1, :] ** expo, axis=0) / (nsam[i] - 1)
                ) ** (1 / expo)
            av[i, nz] = md
    a = np.min(
        av, axis=0
    )  # minimum distance between trials of different classes (down columns) - 1 x nspikes
    for i in range(dists.shape[1]):
        icla = np.max(
            np.where(samstart <= i + 1)
        )  # find the class of the spike-train[i]
        indx = np.where(av[:, i] == a[i])[
            0
        ]  # find the class of the nearest spike-train to spike-train[i]
        if len(indx) > 0:
            anear[icla, indx[0]] += 1 / len(indx)
    return anear.astype(int)
