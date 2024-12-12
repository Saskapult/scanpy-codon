from python import pandas as pd
from python import numpy as np
from python import scipy.sparse
from python import anndata


def _check_indices(
    dim_df,
    alt_index,
    dim: str,
    keys: List[str],
    alias_index = None,
    use_raw: bool = False,
):
    """Common logic for checking indices for obs_df and var_df."""
    alt_repr = "adata.raw" if use_raw else "adata"

    alt_dim = ("obs", "var")[dim == "obs"]

    alias_name = None
    alt_search_repr = "placeholder"
    if alias_index is not None:
        alt_names = pd.Series(alt_index, index=alias_index)
        alias_name = alias_index.name
        alt_search_repr = f"{alt_dim}['{alias_name}']"
    else:
        alt_names = pd.Series(alt_index, index=alt_index)
        alt_search_repr = f"{alt_dim}_names"

    col_keys = []
    index_keys = []
    index_aliases = []
    not_found = []

    # check that adata.obs does not contain duplicated columns
    # if duplicated columns names are present, they will
    # be further duplicated when selecting them.
    if not dim_df.columns.is_unique:
        dup_cols = dim_df.columns[dim_df.columns.duplicated()].tolist()
        raise ValueError(
            f"adata.{dim} contains duplicated columns. Please rename or remove "
            "these columns first.\n`"
            f"Duplicated columns {dup_cols}"
        )

    if not alt_index.is_unique:
        raise ValueError(
            f"{alt_repr}.{alt_dim}_names contains duplicated items\n"
            f"Please rename these {alt_dim} names first for example using "
            f"`adata.{alt_dim}_names_make_unique()`"
        )

    # use only unique keys, otherwise duplicated keys will
    # further duplicate when reordering the keys later in the function
    for key in dict.fromkeys(keys):
        if key in dim_df.columns:
            col_keys.append(key)
            if key in alt_names.index:
                raise KeyError(
                    f"The key '{key}' is found in both adata.{dim} and {alt_repr}.{alt_search_repr}."
                )
        elif key in alt_names.index:
            val = alt_names[key]
            if isinstance(val, pd.Series):
                # while var_names must be unique, adata.var[gene_symbols] does not
                # It's still ambiguous to refer to a duplicated entry though.
                assert alias_index is not None
                raise KeyError(
                    f"Found duplicate entries for '{key}' in {alt_repr}.{alt_search_repr}."
                )
            index_keys.append(val)
            index_aliases.append(key)
        else:
            not_found.append(key)
    if len(not_found) > 0:
        raise KeyError(
            f"Could not find keys '{not_found}' in columns of `adata.{dim}` or in"
            f" {alt_repr}.{alt_search_repr}."
        )

    return col_keys, index_keys, index_aliases



def _get_array_values(
    X,
    dim_names,
    keys: List[str],
    axis: int,
    backed: bool,
):
    # TODO: This should be made easier on the anndata side
    mutable_idxer = [slice(None), slice(None)]
    idx = dim_names.get_indexer(keys)

    # for backed AnnData is important that the indices are ordered
    if backed:
        idx_order = np.argsort(idx)
        rev_idxer = mutable_idxer.copy()
        mutable_idxer[axis] = idx[idx_order]
        rev_idxer[axis] = np.argsort(idx_order)
        matrix = X[tuple(mutable_idxer)][tuple(rev_idxer)]
    else:
        mutable_idxer[axis] = idx
        matrix = X[tuple(mutable_idxer)]

    from python import scipy.sparse

    if sparse.issparse(matrix):
        matrix = matrix.toarray()

    return matrix



def _get_obs_rep(
    adata,
    use_raw: bool = False,
    layer: str = None,
    obsm: str = None,
    obsp: str = None,
):
    """
    Choose array aligned with obs annotation.
    """
    # https://github.com/scverse/scanpy/issues/1546
    if not isinstance(use_raw, bool):
        raise TypeError(f"use_raw expected to be bool, was {type(use_raw)}.")

    is_layer = layer is not None
    is_raw = use_raw is not False
    is_obsm = obsm is not None
    is_obsp = obsp is not None
    choices_made = sum((is_layer, is_raw, is_obsm, is_obsp))
    assert choices_made in {0, 1}
    if choices_made == 0:
        return adata.X
    if is_layer:
        return adata.layers[layer]
    if use_raw:
        return adata.raw.X
    if is_obsm:
        return adata.obsm[obsm]
    if is_obsp:
        return adata.obsp[obsp]
    raise AssertionError(
        "That was unexpected. Please report this bug at:\n\n\t"
        "https://github.com/scverse/scanpy/issues"
    )


def obs_df(
    adata,
    keys: List[str] = [],
    obsm_keys: List[tuple[str, int]] = [],
    layer: str = None,
    gene_symbols: str = None,
    use_raw: bool = False,
):
    if use_raw:
        assert (
            layer is None
        ), "Cannot specify use_raw=True and a layer at the same time."
        var = adata.raw.var
    else:
        var = adata.var
    alias_index = pd.Index(var[gene_symbols]) if gene_symbols is not None else None

    obs_cols, var_idx_keys, var_symbols = _check_indices(
        adata.obs,
        var.index,
        dim="obs",
        keys=keys,
        alias_index=alias_index,
        use_raw=use_raw,
    )

    # Make df
    df = pd.DataFrame(index=adata.obs_names)

    # add var values
    if len(var_idx_keys) > 0:
        matrix = _get_array_values(
            _get_obs_rep(adata, layer=layer, use_raw=use_raw),
            var.index,
            var_idx_keys,
            axis=1,
            backed=adata.isbacked,
        )
        df = pd.concat(
            [df, pd.DataFrame(matrix, columns=var_symbols, index=adata.obs_names)],
            axis=1,
        )

    # add obs values
    if len(obs_cols) > 0:
        df = pd.concat([df, adata.obs[obs_cols]], axis=1)

    # reorder columns to given order (including duplicates keys if present)
    if keys:
        df = df[keys]

    for k, idx in obsm_keys:
        added_k = f"{k}-{idx}"
        val = adata.obsm[k]
        if isinstance(val, np.ndarray):
            df[added_k] = np.ravel(val[:, idx])
        elif isinstance(val, sparse.spmatrix):
            df[added_k] = np.ravel(val[:, idx].toarray())
        elif isinstance(val, pd.DataFrame):
            df[added_k] = val.loc[:, idx]

    return df


def _check_mask(
    data,
    mask,
    dim,
):
    if isinstance(mask, str):
        if not isinstance(data, anndata.AnnData):
            msg = "Cannot refer to mask with string without providing anndata object as argument"
            raise ValueError(msg)

        annot = getattr(data, dim)
        if mask not in annot.columns:
            msg = (
                f"Did not find `adata.{dim}[{mask}]`. "
                f"Either add the mask first to `adata.{dim}`"
                "or consider using the mask argument with a boolean array."
            )
            raise ValueError(msg)
        mask_array = annot[mask].to_numpy()
    else:
        if len(mask) != data.shape[0 if dim == "obs" else 1]:
            raise ValueError("The shape of the mask do not match the data.")
        mask_array = mask

    if not pd.api.types.is_bool_dtype(mask_array.dtype):
        raise ValueError("Mask array must be boolean.")

    return mask_array
