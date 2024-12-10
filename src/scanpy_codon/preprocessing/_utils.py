from python import numba
from python import numpy as np
from python import scipy.sparse
from python import sklearn.random_projection
from python import dask.array as da

from .._utils import axis_sum, elem_mul


def axis_mean(X, axis: int, dtype):# -> DaskArray:
	# Not using single dispatch anymore :(
	if isinstance(X, da.DaskArray):
		total = axis_sum(X, axis=axis, dtype=dtype)
		return total / X.shape[axis]
	else: # np array
		return X.mean(axis=axis, dtype=dtype)


def sparse_mean_var_minor_axis(
    data, indices, indptr, major_len, minor_len, n_threads
):
    rows = len(indptr) - 1
    sums_minor = np.zeros((n_threads, minor_len))
    squared_sums_minor = np.zeros((n_threads, minor_len))
    means = np.zeros(minor_len)
    variances = np.zeros(minor_len)
    for i in numba.prange(n_threads):
        for r in range(i, rows, n_threads):
            for j in range(indptr[r], indptr[r + 1]):
                minor_index = indices[j]
                if minor_index >= minor_len:
                    continue
                value = data[j]
                sums_minor[i, minor_index] += value
                squared_sums_minor[i, minor_index] += value * value
    for c in numba.prange(minor_len):
        sum_minor = sums_minor[:, c].sum()
        means[c] = sum_minor / major_len
        variances[c] = (
            squared_sums_minor[:, c].sum() / major_len - (sum_minor / major_len) ** 2
        )
    return means, variances


def sparse_mean_var_major_axis(data, indptr, major_len, minor_len, n_threads):
    rows = len(indptr) - 1
    means = np.zeros(major_len)
    variances = np.zeros_like(means)

    for i in numba.prange(n_threads):
        for r in range(i, rows, n_threads):
            sum_major = 0.0
            squared_sum_minor = 0.0
            for j in range(indptr[r], indptr[r + 1]):
                value = np.float64(data[j])
                sum_major += value
                squared_sum_minor += value * value
            means[r] = sum_major
            variances[r] = squared_sum_minor
    for c in numba.prange(major_len):
        mean = means[c] / minor_len
        means[c] = mean
        variances[c] = variances[c] / minor_len - mean * mean
    return means, variances


def sparse_mean_variance_axis(mtx, axis: int):
    assert axis in (0, 1)
    if isinstance(mtx, sparse.csr_matrix):
        ax_minor = 1
        shape = mtx.shape
    elif isinstance(mtx, sparse.csc_matrix):
        ax_minor = 0
        shape = mtx.shape[::-1]
    else:
        raise ValueError("This function only works on sparse csr and csc matrices")
    if axis == ax_minor:
        return sparse_mean_var_major_axis(
            mtx.data,
            mtx.indptr,
            major_len=shape[0],
            minor_len=shape[1],
            n_threads=numba.get_num_threads(),
        )
    else:
        return sparse_mean_var_minor_axis(
            mtx.data,
            mtx.indices,
            mtx.indptr,
            major_len=shape[0],
            minor_len=shape[1],
            n_threads=numba.get_num_threads(),
        )


def _get_mean_var(
    X, axis: int = 0
):# -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if isinstance(X, sparse.spmatrix):
        mean, var = sparse_mean_variance_axis(X, axis=axis)
    else:
        mean = axis_mean(X, axis=axis, dtype=np.float64)
        mean_sq = axis_mean(elem_mul(X, X), axis=axis, dtype=np.float64)
        var = mean_sq - mean**2
    # enforce R convention (unbiased estimator) for variance
    if X.shape[axis] != 1:
        var *= X.shape[axis] / (X.shape[axis] - 1)
    return mean, var


def sample_comb(
    dims,
    nsamp: int,
    random_state = None,
    method: str = "auto",
):# -> NDArray[np.int64]:
    idx = random_projection.sample_without_replacement(
        np.prod(dims), nsamp, random_state=random_state, method=method
    )
    return np.vstack(np.unravel_index(idx, dims)).T


def _to_dense_csc_numba(
    indptr,
    indices,
    data,
    X,
    shape: tuple[int, int],
):# -> None:
    for c in numba.prange(X.shape[1]):
        for i in range(indptr[c], indptr[c + 1]):
            X[indices[i], c] = data[i]


def _to_dense_csr_numba(
    indptr,
    indices,
    data,
    X,
    shape: tuple[int, int],
):# -> None:
    for r in numba.prange(shape[0]):
        for i in range(indptr[r], indptr[r + 1]):
            X[r, indices[i]] = data[i]


def _to_dense(
    X,
    order: str = "C",
):# -> NDArray:
    out = np.zeros(X.shape, dtype=X.dtype, order=order)
    if X.format == "csr":
        _to_dense_csr_numba(X.indptr, X.indices, X.data, out, X.shape)
    elif X.format == "csc":
        _to_dense_csc_numba(X.indptr, X.indices, X.data, out, X.shape)
    else:
        out = X.toarray(order=order)
    return out



