from python import numpy as np
from python import dask.array as da
from python import scipy.sparse


def axis_nnz(X, axis: int = 0):
    return np.count_nonzero(X, axis=axis)


def axis_sum(
    X,
    axis: int = None,
    dtype = None,
):
    return np.sum(X, axis=axis, dtype=dtype)


def elem_mul(x, y):
    if (isinstance(x, np.ndarray) or isinstance(x, sparse.spmatrix)) and (isinstance(y, np.ndarray) or isinstance(y, sparse.spmatrix)):
        if isinstance(x, sparse.spmatrix):
            return type(x)(x.multiply(y))
        else:
            return x * y
    # elif isinstance(x, da.DaskArray) and isinstance(x, da.DaskArray):
    return da.map_blocks(elem_mul, x, y)
    # else:
    #     raise NotImplementedError


def _check_use_raw(
    adata, use_raw: bool, layer: str = None
) -> bool:
    if use_raw is not None:
        return use_raw
    if layer is not None:
        return False
    return adata.raw is not None
