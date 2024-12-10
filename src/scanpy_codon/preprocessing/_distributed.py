from python import dask.array as da
from python import numpy as np


def materialize_as_ndarray(a):
    """Compute distributed arrays and convert them to numpy ndarrays."""
    if isinstance(a, da.DaskArray):
        return a.compute()
    if not isinstance(a, tuple):
        return np.asarray(a)

    if not any(isinstance(arr, da.DaskArray) for arr in a):
        return tuple(np.asarray(arr) for arr in a)

    return da.compute(*a, sync=True)
