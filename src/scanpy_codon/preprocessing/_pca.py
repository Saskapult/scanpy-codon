from ..get import _check_mask, _get_obs_rep
from python import warnings
from python import anndata
from python import dask.array as da
from python import sklearn.utils
from python import scipy.sparse


def _handle_mask_var(
    adata,
    mask_var,
    use_highly_variable,
):
    """\
    Unify new mask argument and deprecated use_highly_varible argument.

    Returns both the normalized mask parameter and the validated mask array.
    """
    # First, verify and possibly warn
    if use_highly_variable is not None:
        hint = (
            'Use_highly_variable=True can be called through mask_var="highly_variable". '
            "Use_highly_variable=False can be called through mask_var=None"
        )
        msg = f"Argument `use_highly_variable` is deprecated, consider using the mask argument. {hint}"
        warnings.warn(msg)
        if mask_var is not None:
            msg = f"These arguments are incompatible. {hint}"
            raise ValueError(msg)

    # Handle default case and explicit use_highly_variable=True
    if use_highly_variable or (
        use_highly_variable is None
        and mask_var is None
        and "highly_variable" in adata.var.columns
    ):
        mask_var = "highly_variable"

    # Without highly variable genes, we don’t use a mask by default
    if mask_var is None or mask_var is None:
        return None, None
    return mask_var, _check_mask(adata, mask_var, "var")


def pca(
    data,
    n_comps: int = None,
    layer: str = None,
    zero_center: bool = True,
    svd_solver = None,
    random_state = 0,
    return_info: bool = False,
    mask_var = None,
    use_highly_variable: bool = None,
    dtype = "float32",
    chunked: bool = False,
    chunk_size: int = None,
    key_added: str = None,
    copy: bool = False,
):
    # logg_start = logg.info("computing PCA")
    if layer is not None and chunked:
        # Current chunking implementation relies on pca being called on X
        raise NotImplementedError("Cannot use `layer` and `chunked` at the same time.")

    # chunked calculation is not randomized, anyways
    # if svd_solver in {"auto", "randomized"} and not chunked:
    #     logg.info(
    #         "Note that scikit-learn's randomized PCA might not be exactly "
    #         "reproducible across different computational platforms. For exact "
    #         "reproducibility, choose `svd_solver='arpack'`."
    #     )
    data_is_AnnData = isinstance(data, anndata.AnnData)
    if data_is_AnnData:
        if layer is None and not chunked:
            raise NotImplementedError(
                f"PCA is not implemented for matrices of type {type(data.X)} with chunked as False"
            )
        adata = data.copy() if copy else data
    else:
        # if pkg_version("anndata") < Version("0.8.0rc1"):
        #     adata = anndata.AnnData(data, dtype=data.dtype)
        # else:
        adata = anndata.AnnData(data)

    # Unify new mask argument and deprecated use_highly_varible argument
    mask_var_param, mask_var = _handle_mask_var(adata, mask_var, use_highly_variable)
    # del use_highly_variable
    adata_comp = adata[:, mask_var] if mask_var is not None else adata

    if n_comps is None:
        min_dim = min(adata_comp.n_vars, adata_comp.n_obs)
        n_comps = min_dim - 1 if min_dim <= 50 else 50

    # logg.info(f"    with n_comps={n_comps}")

    X = _get_obs_rep(adata_comp, layer=layer)
    # if is_backed_type(X) and layer is not None:
    #     raise NotImplementedError(
    #         f"PCA is not implemented for matrices of type {type(X)} from layers"
    #     )
    # # See: https://github.com/scverse/scanpy/pull/2816#issuecomment-1932650529
    # if (
    #     Version(ad.__version__) < Version("0.9")
    #     and mask_var is not None
    #     and isinstance(X, np.ndarray)
    # ):
    #     warnings.warn(
    #         "When using a mask parameter with anndata<0.9 on a dense array, the PCA"
    #         "can have slightly different results due the array being column major "
    #         "instead of row major.",
    #     )

    # check_random_state returns a numpy RandomState when passed an int but
    # dask needs an int for random state
    if not isinstance(X, da.Array):
        random_state = utils.check_random_state(random_state)
    elif not isinstance(random_state, int):
        msg = f"random_state needs to be an int, not a {type(random_state).__name__} when passing a dask array"
        raise TypeError(msg)

    if chunked:
        # if (
        #     not zero_center
        #     or random_state
        #     or (svd_solver is not None and svd_solver != "arpack")
        # ):
        #     logg.debug("Ignoring zero_center, random_state, svd_solver")

        incremental_pca_kwargs = dict()
        # if isinstance(X, da.Array):
        #     from dask.array import zeros
        #     from dask_ml.decomposition import IncrementalPCA

        #     incremental_pca_kwargs["svd_solver"] = _handle_dask_ml_args(
        #         svd_solver, IncrementalPCA
        #     )
        # else:
        print("Assumign this is not a daskarray")
        from python import numpy as np 
        from python import sklearn.decomposition

        X_pca = np.zeros((X.shape[0], n_comps), X.dtype)

        pca_ = decomposition.IncrementalPCA(n_components=n_comps, **incremental_pca_kwargs)

        for chunk, _, _ in adata_comp.chunked_X(chunk_size):
            chunk = chunk.toarray() if sparse.issparse(chunk) else chunk
            pca_.partial_fit(chunk)

        for chunk, start, end in adata_comp.chunked_X(chunk_size):
            chunk = chunk.toarray() if sparse.issparse(chunk) else chunk
            X_pca[start:end] = pca_.transform(chunk)
    elif zero_center:
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
        if not isinstance(X, da.Array):
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
                svd_solver=svd_solver,
                random_state=random_state,
            )
        elif sparse.issparse(X._meta):
            # from ._dask_sparse import PCASparseDask

            # if random_state != 0:
            #     msg = f"Ignoring {random_state=} when using a sparse dask array"
            #     warnings.warn(msg)
            # if svd_solver not in {None, "covariance_eigh"}:
            #     msg = f"Ignoring {svd_solver=} when using a sparse dask array"
            #     warnings.warn(msg)
            # pca_ = PCASparseDask(n_components=n_comps)
            print("Unimplemented part")
            pca_ = None
        else:
            # from dask_ml.decomposition import PCA

            # svd_solver = _handle_dask_ml_args(svd_solver, PCA)
            # pca_ = PCA(
            #     n_components=n_comps,
            #     svd_solver=svd_solver,
            #     random_state=random_state,
            # )
            print("Unimplemented part")
            pca_ = None
        X_pca = pca_.fit_transform(X)

    # See _handle_sklearn_args confusion
    else:
        print("Trying lack of argument handling")
        # if isinstance(X, da.Array):
        #     if sparse.issparse(X._meta):
        #         msg = "Dask sparse arrays do not support zero-centering (yet)"
        #         raise TypeError(msg)
        #     from python import dask_ml.decomposition

        #     svd_solver = _handle_dask_ml_args(svd_solver, decomposition.TruncatedSVD)
        # else:
        #     from python import sklearn.decomposition

        #     svd_solver = _handle_sklearn_args(svd_solver, decomposition.TruncatedSVD)

        # logg.debug(
        #     "    without zero-centering: \n"
        #     "    the explained variance does not correspond to the exact statistical definition\n"
        #     "    the first component, e.g., might be heavily influenced by different means\n"
        #     "    the following components often resemble the exact PCA very closely"
        # )

        from python import sklearn.decomposition

        pca_ = decomposition.TruncatedSVD(
            n_components=n_comps, random_state=random_state, algorithm=svd_solver
        )
        X_pca = pca_.fit_transform(X)

    if X_pca.dtype.descr != np.dtype(dtype).descr:
        X_pca = X_pca.astype(dtype)

    if data_is_AnnData:
        key_obsm, key_varm, key_uns = (
            ("X_pca", "PCs", "pca") if key_added is None else [key_added] * 3
        )
        adata.obsm[key_obsm] = X_pca

        if mask_var is not None:
            adata.varm[key_varm] = np.zeros(shape=(adata.n_vars, n_comps))
            adata.varm[key_varm][mask_var] = pca_.components_.T
        else:
            adata.varm[key_varm] = pca_.components_.T

        params = dict(
            zero_center=zero_center,
            use_highly_variable=mask_var_param == "highly_variable",
            mask_var=mask_var_param,
        )
        if layer is not None:
            params["layer"] = layer
        adata.uns[key_uns] = dict(
            params=params,
            variance=pca_.explained_variance_,
            variance_ratio=pca_.explained_variance_ratio_,
        )

        # logg.info("    finished", time=logg_start)
        # logg.debug(
        #     "and added\n"
        #     f"    {key_obsm!r}, the PCA coordinates (adata.obs)\n"
        #     f"    {key_varm!r}, the loadings (adata.varm)\n"
        #     f"    'pca_variance', the variance / eigenvalues (adata.uns[{key_uns!r}])\n"
        #     f"    'pca_variance_ratio', the variance ratio (adata.uns[{key_uns!r}])"
        # )
        return adata if copy else None
    else:
        # logg.info("    finished", time=logg_start)
        if return_info:
            return (
                X_pca,
                pca_.components_,
                pca_.explained_variance_ratio_,
                pca_.explained_variance_,
            )
        else:
            return X_pca
