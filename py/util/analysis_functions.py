from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import io as spio
from scipy import stats
from py.metrics import metrics

def correlation(arr1, arr2) -> tuple:
    """Return the correlation between two arrays. Tuple(Correlation, p-value)"""
    return stats.pearsonr(arr1, arr2)

def within_labels(arr, discrete=True):
    sequential_count = 0
    previous_value = None
    result, event = [], []
    for index, value in enumerate(arr):
        if value != previous_value:
            if previous_value is not None:
                middle_index = index - (sequential_count // 2)
                result.append(middle_index)
                event.append(arr[index - 1])
                sequential_count = 0
            previous_value = value
        sequential_count += 1

        if index == len(arr) - 1:
            event.append(arr[-1])
            middle_index = index - (sequential_count // 2)
            if discrete: # for arrays without middle values
                result.append(middle_index + 1)
            else:
                result.append(middle_index)

    return result, event

def between_labels(arr):
    previous_value = None
    result, events = [], []
    for index, value in enumerate(arr):
        if value != previous_value:
            if previous_value is not None:
                result.append(index)
            previous_value = value
    return result

def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_keys(d):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], spio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _has_struct(elem):
        """Determine if elem is an array and if any array item is a struct"""
        return isinstance(elem, np.ndarray) and any(isinstance(
            e, spio.matlab.mat_struct) for e in elem)

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif _has_struct(elem):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        """
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif _has_struct(sub_elem):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def get_predictions_arr(cm):
    """Reverse engineer y_true and y_pred from a confusion matrix."""
    y_true = []
    y_pred = []
    cm = cm.astype(int)
    for i in range(cm.shape[0]):  # For each class
        for j in range(cm.shape[1]):  # For each prediction
            y_true.extend([i] * cm[i, j])
            y_pred.extend([j] * cm[i, j])

    return y_true, y_pred

def get_stats(data, excel=False) -> tuple:
    conf = data["anear"]
    n = conf.shape[0] // 2
    conf_top_right = conf[:n, n:, :]
    conf_bottom_left = conf[n:, :n, :]

    qvals = data["qvals"]
    tr_info = data["tr_info"]
    tr_ja = data["tr_ja"]
    tr_tp = data["tr_tp"]

    bl_info = data["bl_info"]
    bl_ja = data["bl_ja"]
    bl_tp = data["bl_tp"]

    bl_acc = [metrics.accuracy(conf_bottom_left[:, :, i]) for i in range(qvals.shape[0])]
    tr_acc = [metrics.accuracy(conf_top_right[:, :, i]) for i in range(qvals.shape[0])]
    bl_ja_corr = correlation(bl_acc, bl_ja)
    tr_ja_corr = correlation(tr_acc, tr_ja)
    bl_tp_corr = correlation(bl_acc, bl_tp)
    tr_tp_corr = correlation(tr_acc, tr_tp)
    bl_info_corr = correlation(bl_acc, bl_info)
    tr_info_corr = correlation(tr_acc, tr_info)

    corr = pd.DataFrame(
        {
            "bl_ja": bl_ja_corr,
            "tr_ja": tr_ja_corr,
            "bl_tp": bl_tp_corr,
            "tr_tp": tr_tp_corr,
            "bl_info": bl_info_corr,
            "tr_info": tr_info_corr,
        }
    )

    if excel:
        corr.to_excel(r"C:\Users\Flynn\OneDrive\Desktop\temp\corr2.xlsx")

    stat = pd.DataFrame(
        {
            "q": qvals,
            "bl_acc": bl_acc,
            "tr_acc": tr_acc,
            "bl_ja": bl_ja,
            "tr_ja": tr_ja,
            "bl_tp": bl_tp,
            "tr_tp": tr_tp,
            "bl_info": bl_info,
            "tr_info": tr_info,
        }
    )
    if excel:
        stat.to_excel(r"C:\Users\Flynn\OneDrive\Desktop\temp\stat2.xlsx")

    return corr.round(decimals=3), stat.round(decimals=3)

def validate_matrix_nsam(matrix, nsam):
    """Validate that matrix has nsam samples."""
    results = []
    for q in range(nsam):
        # skip q = 0 because there are rounding issues
        # TODO: fix rounding issues
        if q != 0:
            # Extract the 2D slice at each q value
            slice_q = matrix[:, :, q]

            # Check if each row in the slice sums to the corresponding value in the target array
            is_correct = np.all([np.sum(slice_q[i]) == nsam[i] for i in range(slice_q.shape[0])])

            # Append the result to the results list
            results.append(is_correct)

    # Return True only if all results are True
    return np.all(results)