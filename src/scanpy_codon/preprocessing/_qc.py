from python import anndata
# from python import scipy.sparse.csr_matrix, scipy.sparse.issparse, scipy.sparse.isspmatrix_coo, scipy.sparse.isspmatrix_csr, scipy.sparse.spmatrix
from python import scipy.sparse
from python import pandas as pd
from python import numpy as np

from _distributed import materialize_as_ndarray
from _utils import _get_mean_var
from .._utils import axis_nnz, axis_sum


def _choose_mtx_rep(adata, use_raw: bool = False, layer: str = ""):
	if layer != "":
	    return adata.layers[layer]
	elif use_raw:
		return adata.raw.X
	else:
		return adata.X


def top_segment_proportions(mtx, ns: List[int]):# -> np.ndarray:
    # Currently ns is considered to be 1 indexed
    ns = np.sort(ns)
    sums = mtx.sum(axis=1)
    partitioned = np.apply_along_axis(np.partition, 1, mtx, mtx.shape[1] - ns)[:, ::-1][
        :, : ns[-1]
    ]
    values = np.zeros((mtx.shape[0], len(ns)))
    acc = np.zeros(mtx.shape[0])
    prev = 0
    for j, n in enumerate(ns):
        acc += partitioned[:, prev:n].sum(axis=1)
        values[:, j] = acc
        prev = n
    return values / sums[:, None]


def describe_obs(
    adata,
    expr_type: str = "counts",
    var_type: str = "genes",
    qc_vars: List[str] = (),
    percent_top: List[int] = (50, 100, 200, 500),
    layer: str = "",
    use_raw: bool = False,
    log1p: bool = True,
    inplace: bool = False,
    X=None,
):# -> pd.DataFrame | None:
    # Handle whether X is passed
    if X is None:
        X = _choose_mtx_rep(adata, use_raw=use_raw, layer=layer)
        if sparse.isspmatrix_coo(X):
            X = sparse.csr_matrix(X)  # COO not subscriptable
        if sparse.issparse(X):
            X.eliminate_zeros()
    obs_metrics = pd.DataFrame(index=adata.obs_names)
    obs_metrics[f"n_{var_type}_by_{expr_type}"] = materialize_as_ndarray(
        axis_nnz(X, axis=1)
    )
    if log1p:
        obs_metrics[f"log1p_n_{var_type}_by_{expr_type}"] = np.log1p(
            obs_metrics[f"n_{var_type}_by_{expr_type}"]
        )
    obs_metrics[f"total_{expr_type}"] = np.ravel(axis_sum(X, axis=1))
    if log1p:
        obs_metrics[f"log1p_total_{expr_type}"] = np.log1p(
            obs_metrics[f"total_{expr_type}"]
        )
    if percent_top:
        percent_top = sorted(percent_top)
        proportions = top_segment_proportions(X, percent_top)
        for i, n in enumerate(percent_top):
            obs_metrics[f"pct_{expr_type}_in_top_{n}_{var_type}"] = (
                proportions[:, i] * 100
            )
    for qc_var in qc_vars:
        obs_metrics[f"total_{expr_type}_{qc_var}"] = np.ravel(
            axis_sum(X[:, adata.var[qc_var].values], axis=1)
        )
        if log1p:
            obs_metrics[f"log1p_total_{expr_type}_{qc_var}"] = np.log1p(
                obs_metrics[f"total_{expr_type}_{qc_var}"]
            )
        obs_metrics[f"pct_{expr_type}_{qc_var}"] = (
            obs_metrics[f"total_{expr_type}_{qc_var}"]
            / obs_metrics[f"total_{expr_type}"]
            * 100
        )
    if inplace:
        adata.obs[obs_metrics.columns] = obs_metrics
    else:
        return obs_metrics
    return None


@python
def describe_var(
    adata,
    expr_type: str = "counts",
    var_type: str = "genes",
    layer: str = "",
    use_raw: bool = False,
    inplace: bool = False,
    log1p: bool = True,
    X = None,
):# -> pd.DataFrame | None:
    # Handle whether X is passed
    if X is None:
        X = _choose_mtx_rep(adata, use_raw=use_raw, layer=layer)
        if sparse.isspmatrix_coo(X):
            X = sparse.csr_matrix(X)  # COO not subscriptable
        if sparse.issparse(X):
            X.eliminate_zeros()
    var_metrics = pd.DataFrame(index=adata.var_names)
    var_metrics["n_cells_by_{expr_type}"], var_metrics["mean_{expr_type}"] = (
        materialize_as_ndarray((axis_nnz(X, axis=0), _get_mean_var(X, axis=0)[0]))
    )
    if log1p:
        var_metrics["log1p_mean_{expr_type}"] = np.log1p(
            var_metrics["mean_{expr_type}"]
        )
    var_metrics["pct_dropout_by_{expr_type}"] = (
        1 - var_metrics["n_cells_by_{expr_type}"] / X.shape[0]
    ) * 100
    var_metrics["total_{expr_type}"] = np.ravel(axis_sum(X, axis=0))
    if log1p:
        var_metrics["log1p_total_{expr_type}"] = np.log1p(
            var_metrics["total_{expr_type}"]
        )
    # Relabel
    new_colnames = []
    for col in var_metrics.columns:
        # Codon has not locals function it seems 
        new_colnames.append(col.format(**locals()))
    var_metrics.columns = new_colnames
    if inplace:
        adata.var[var_metrics.columns] = var_metrics
        return None
    return var_metrics



def calculate_qc_metrics(
    adata,
    expr_type: str = "counts",
    var_type: str = "genes",
    qc_vars: List[str] = (),
    percent_top: List[int] = [50, 100, 200, 500],
    layer: str = "",
    use_raw: bool = False,
    inplace: bool = False,
    log1p: bool = True,
):# -> tuple[pd.DataFrame, pd.DataFrame] | None:
    # Pass X so I only have to do it once
    X = _choose_mtx_rep(adata, use_raw=use_raw, layer=layer)
    if sparse.isspmatrix_coo(X):
        X = sparse.csr_matrix(X)  # COO not subscriptable
    if sparse.issparse(X):
        X.eliminate_zeros()

    obs_metrics = describe_obs(
        adata,
        expr_type=expr_type,
        var_type=var_type,
        qc_vars=qc_vars,
        percent_top=percent_top,
        inplace=inplace,
        X=X,
        log1p=log1p,
    )
    var_metrics = describe_var(
        adata,
        expr_type=expr_type,
        var_type=var_type,
        inplace=inplace,
        X=X,
        log1p=log1p,
    )

    if not inplace:
        return obs_metrics, var_metrics

