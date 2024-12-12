from ..get import _check_mask, _get_obs_rep
from python import warnings
from python import anndata
from python import dask.array as da
from python import sklearn.utils
from python import scipy.sparse
from python import numpy as np


def _handle_mask_var(
    adata,
    mask_var: str = None,
):
    if mask_var is None and "highly_variable" in adata.var.columns:
        mask_var = "highly_variable"

    # Without highly variable genes, we don’t use a mask by default
    if mask_var is None or mask_var is None:
        return None, None
    return mask_var, _check_mask(adata, mask_var, "var")


# @python
def pca(
    data,
):
    random_state: int = 0
    # mask_var: str = None
    dtype: str = "float32"
    chunked: bool = False
    copy: bool = False

    adata = data

    # Unify new mask argument and deprecated use_highly_varible argument
    # ggg = _handle_mask_var(adata, mask_var=None)
    # mask_var_param: str = ggg[0] 
    # mask_var: str = ggg[1]
    # del use_highly_variable
    # adata_comp = adata[:, mask_var] if mask_var is not None else adata
    adata_comp = adata

    
    min_dim = min(adata_comp.n_vars, adata_comp.n_obs)
    n_comps = min_dim - 1 if min_dim <= 50 else 50

    X: pyobj = _get_obs_rep(adata_comp)

    # check_random_state returns a numpy RandomState when passed an int but
    # dask needs an int for random state
    # if not isinstance(X, da.Array):
    random_state = utils.check_random_state(random_state)
    # elif not isinstance(random_state, int):
    #     msg = f"random_state needs to be an int, not a {type(random_state).__name__} when passing a dask array"
    #     raise TypeError(msg)

    # if sparse.issparse(X) and (
    #     pkg_version("scikit-learn") < Version("1.4") or svd_solver == "lobpcg"
    # ):
    #     if svd_solver not in (
    #         {"lobpcg"} | get_literal_vals(SvdSolvPCASparseSklearn)
    #     ):
    #         if svd_solver is not None:
    #             msg = (
    #                 f"Ignoring {svd_solver=} and using 'arpack', "
    #                 "sparse PCA with sklearn < 1.4 only supports 'lobpcg' and 'arpack'."
    #             )
    #             warnings.warn(msg)
    #         svd_solver = "arpack"
    #     elif svd_solver == "lobpcg":
    #         msg = (
    #             f"{svd_solver=} for sparse relies on legacy code and will not be supported in the future. "
    #             "Also the lobpcg solver has been observed to be inaccurate. Please use 'arpack' instead."
    #         )
    #         warnings.warn(msg)
    #     X_pca, pca_ = _pca_compat_sparse(
    #         X, n_comps, solver=svd_solver, random_state=random_state
    #     )
    # else:
    # if not isinstance(X, da.Array):
    from python import sklearn.decomposition

    # svd_solver = _handle_sklearn_args(svd_solver, PCA, sparse=sparse.issparse(X))
    # ^^^ idk what this does, tying without it
    print("Trying lack of argument handling")
    # if sparse SvdSolvPCASparseSklearn, default arpack
    # else SvdSolvPCADenseSklearn, default arpack
    # get literal values of that 
    # return svd solver? 
    pca_ = decomposition.PCA(
        n_components=n_comps,
        # svd_solver=svd_solver,
        random_state=random_state,
    )
    # elif sparse.issparse(X._meta):
    #     # from ._dask_sparse import PCASparseDask

    #     # if random_state != 0:
    #     #     msg = f"Ignoring {random_state=} when using a sparse dask array"
    #     #     warnings.warn(msg)
    #     # if svd_solver not in {None, "covariance_eigh"}:
    #     #     msg = f"Ignoring {svd_solver=} when using a sparse dask array"
    #     #     warnings.warn(msg)
    #     # pca_ = PCASparseDask(n_components=n_comps)
    #     print("Unimplemented part")
    #     pca_ = None
    # else:
    #     # from dask_ml.decomposition import PCA

    #     # svd_solver = _handle_dask_ml_args(svd_solver, PCA)
    #     # pca_ = PCA(
    #     #     n_components=n_comps,
    #     #     svd_solver=svd_solver,
    #     #     random_state=random_state,
    #     # )
    #     print("Unimplemented part")
    #     pca_ = None
    X_pca: pyobj = pca_.fit_transform(X)


    # from python import sklearn.decomposition

    # pca_ = decomposition.TruncatedSVD(
    #     n_components=n_comps, random_state=random_state, algorithm=svd_solver
    # )
    # X_pca = pca_.fit_transform(X)

    if X_pca.dtype.descr != np.dtype(dtype).descr:
        X_pca = X_pca.astype(dtype)

    key_obsm = "X_pca"
    key_varm = "PCs"
    key_uns = "pca"
    adata.obsm[key_obsm] = X_pca

    adata.varm[key_varm] = pca_.components_.T

    # params = dict(
    #     zero_center=zero_center,
    #     use_highly_variable=mask_var_param == "highly_variable",
    #     mask_var=mask_var_param,
    # )
    # params = {}
    # params["zero_center"] = zero_center
    # params["use_highly_variable"] = mask_var_param == "highly_variable"
    # # params["mask_var"] = mask_var_param

    # if layer is not None:
    #     params["layer"] = layer
    # adata.uns[key_uns] = dict(
    #     params=params,
    #     variance=pca_.explained_variance_,
    #     variance_ratio=pca_.explained_variance_ratio_,
    # )

    # logg.info("    finished", time=logg_start)
    # logg.debug(
    #     "and added\n"
    #     f"    {key_obsm!r}, the PCA coordinates (adata.obs)\n"
    #     f"    {key_varm!r}, the loadings (adata.varm)\n"
    #     f"    'pca_variance', the variance / eigenvalues (adata.uns[{key_uns!r}])\n"
    #     f"    'pca_variance_ratio', the variance ratio (adata.uns[{key_uns!r}])"
    # )
    return adata if copy else None
