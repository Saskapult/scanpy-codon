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


def _fallback_to_uns(dct, conns, dists, conns_key, dists_key):
    if conns is None and conns_key in dct:
        conns = dct[conns_key]
    if dists is None and dists_key in dct:
        dists = dct[dists_key]

    return conns, dists


class NeighborsView:
    def __init__(self, adata, key=None):
        self._connectivities = None
        self._distances = None

        if key is None or key == "neighbors":
            if "neighbors" not in adata.uns:
                raise KeyError('No "neighbors" in .uns')
            self._neighbors_dict = adata.uns["neighbors"]
            self._conns_key = "connectivities"
            self._dists_key = "distances"
        else:
            if key not in adata.uns:
                raise KeyError(f'No "{key}" in .uns')
            self._neighbors_dict = adata.uns[key]
            self._conns_key = self._neighbors_dict["connectivities_key"]
            self._dists_key = self._neighbors_dict["distances_key"]

        if self._conns_key in adata.obsp:
            self._connectivities = adata.obsp[self._conns_key]
        if self._dists_key in adata.obsp:
            self._distances = adata.obsp[self._dists_key]

        # fallback to uns
        self._connectivities, self._distances = _fallback_to_uns(
            self._neighbors_dict,
            self._connectivities,
            self._distances,
            self._conns_key,
            self._dists_key,
        )

    def __getitem__(self, key: str):
        if key == "distances":
            if "distances" not in self:
                raise KeyError(f'No "{self._dists_key}" in .obsp')
            return self._distances
        elif key == "connectivities":
            if "connectivities" not in self:
                raise KeyError(f'No "{self._conns_key}" in .obsp')
            return self._connectivities
        elif key == "connectivities_key":
            return self._conns_key
        else:
            return self._neighbors_dict[key]

    def __contains__(self, key: str):
        if key == "distances":
            return self._distances is not None
        elif key == "connectivities":
            return self._connectivities is not None
        else:
            return key in self._neighbors_dict


def _choose_graph(adata, obsp, neighbors_key):
    """Choose connectivities from neighbbors or another obsp column"""
    if obsp is not None and neighbors_key is not None:
        raise ValueError(
            "You can't specify both obsp, neighbors_key. " "Please select only one."
        )

    if obsp is not None:
        return adata.obsp[obsp]
    else:
        neighbors = NeighborsView(adata, neighbors_key)
        if "connectivities" not in neighbors:
            raise ValueError(
                "You need to run `pp.neighbors` first "
                "to compute a neighborhood graph."
            )
        return neighbors["connectivities"]
