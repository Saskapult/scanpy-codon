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

# `get_args` returns `tuple[Any]` so I don’t think it’s possible to get the correct type here
@python
def get_literal_vals(typ):
    from types import MethodType, ModuleType, UnionType
    """Get all literal values from a Literal or Union of … of Literal type."""
    if isinstance(typ, UnionType):
        return reduce(
            or_, (dict.fromkeys(get_literal_vals(t)) for t in get_args(typ))
        ).keys()
    if get_origin(typ) is Literal:
        return dict.fromkeys(get_args(typ)).keys()
    msg = f"{typ} is not a valid Literal"
    raise TypeError(msg)
