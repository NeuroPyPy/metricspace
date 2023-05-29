import numpy as np

def to_writeable(source):
    """ Convert input object ``source`` to something we can write

    Parameters
    ----------
    source : object

    Returns
    -------
    arr : None or ndarray or EmptyStructMarker
        If `source` cannot be converted to something we can write to a matfile,
        return None.  If `source` is equivalent to an empty dictionary, return
        ``EmptyStructMarker``.  otherwise return `source` converted to a
        ndarray with contents for writing to matfile.
    """
    if isinstance(source, np.ndarray):
        return source
    if source is None:
        return None
    # Objects that implement mappings
    is_mapping = (hasattr(source, 'keys') and hasattr(source, 'values') and
                  hasattr(source, 'items'))
    # Objects that don't implement mappings, but do have dicts
    if isinstance(source, np.generic):
        # NumPy scalars are never mappings (PyPy issue workaround)
        pass
    elif not is_mapping and hasattr(source, '__dict__'):
        source = dict((key, value) for key, value in source.__dict__.items()
                      if not key.startswith('_'))
        is_mapping = True
    if is_mapping:
        dtype = []
        values = []
        for field, value in source.items():
            if (isinstance(field, str) and
                    field[0] not in '_0123456789'):
                dtype.append((str(field), object))
                values.append(value)
        if dtype:
            return np.array([tuple(values)], dtype)
        else:
            return None
    # Next try and convert to an array
    narr = np.asanyarray(source)
    if narr.dtype.type in (object, np.object_) and \
       narr.shape == () and narr == source:
        # No interesting conversion possible
        return None
    return narr

def write(arr):
    """ Write `arr` to stream at top and sub levels

    Parameters
    ----------
    arr : array_like
        array-like object to create writer for
    """
    # Try to convert things that aren't arrays
    narr = to_writeable(arr)
    if narr is None:
        raise TypeError('Could not convert %s (type %s) to array'
                        % (arr, type(arr)))

    write_cells(narr)

def write_cells(arr):
    # loop over data, column major
    A = np.atleast_2d(arr).flatten('F')
    for el in A:
        write(el)

