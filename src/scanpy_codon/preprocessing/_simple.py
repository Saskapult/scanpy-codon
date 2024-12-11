from ._distributed import materialize_as_ndarray
from python import anndata
from python import scipy.sparse
from .._utils import axis_nnz, axis_sum


def filter_cells(
    data,
    min_counts: int = None,
    min_genes: int = None,
    max_counts: int = None,
    max_genes: int = None,
    inplace: bool = True,
    copy: bool = False, # deprecated 
):
    n_given_options = sum(
        option is not None for option in [min_genes, min_counts, max_genes, max_counts]
    )
    if n_given_options != 1:
        raise ValueError(
            "Only provide one of the optional parameters `min_counts`, "
            "`min_genes`, `max_counts`, `max_genes` per call."
        )
    if isinstance(data, anndata.AnnData):
        # raise_not_implemented_error_if_backed_type(data.X, "filter_cells")
        adata = data.copy() if copy else data
        cell_subset, number = materialize_as_ndarray(
            filter_cells(
                adata.X,
                min_counts=min_counts,
                min_genes=min_genes,
                max_counts=max_counts,
                max_genes=max_genes,
            ),
        )
        if not inplace:
            return cell_subset, number
        if min_genes is None and max_genes is None:
            adata.obs["n_counts"] = number
        else:
            adata.obs["n_genes"] = number
        adata._inplace_subset_obs(cell_subset)
        return adata if copy else None
    X = data  # proceed with processing the data matrix
    min_number = min_counts if min_genes is None else min_genes
    max_number = max_counts if max_genes is None else max_genes
    number_per_cell = axis_sum(
        X if min_genes is None and max_genes is None else X > 0, axis=1
    )
    if sparse.issparse(X):
        number_per_cell = number_per_cell.A1
    if min_number is not None:
        cell_subset = number_per_cell >= min_number
    if max_number is not None:
        cell_subset = number_per_cell <= max_number

    s = axis_sum(~cell_subset)
    if s > 0:
        msg = f"filtered out {s} cells that have "
        if min_genes is not None or min_counts is not None:
            msg += "less than "
            msg += (
                f"{min_genes} genes expressed"
                if min_counts is None
                else f"{min_counts} counts"
            )
        if max_genes is not None or max_counts is not None:
            msg += "more than "
            msg += (
                f"{max_genes} genes expressed"
                if max_counts is None
                else f"{max_counts} counts"
            )
        # logg.info(msg)
    return cell_subset, number_per_cell


def filter_genes(
    data,
    min_counts: int = None,
    min_cells: int = None,
    max_counts: int = None,
    max_cells: int = None,
    inplace: bool = True,
    copy: bool = False,
):
    # if copy:
    #     logg.warning("`copy` is deprecated, use `inplace` instead.")
    n_given_options = sum(
        option is not None for option in [min_cells, min_counts, max_cells, max_counts]
    )
    if n_given_options != 1:
        raise ValueError(
            "Only provide one of the optional parameters `min_counts`, "
            "`min_cells`, `max_counts`, `max_cells` per call."
        )

    if isinstance(data, anndata.AnnData):
        # raise_not_implemented_error_if_backed_type(data.X, "filter_genes")
        adata = data.copy() if copy else data
        gene_subset, number = materialize_as_ndarray(
            filter_genes(
                adata.X,
                min_cells=min_cells,
                min_counts=min_counts,
                max_cells=max_cells,
                max_counts=max_counts,
            )
        )
        if not inplace:
            return gene_subset, number
        if min_cells is None and max_cells is None:
            adata.var["n_counts"] = number
        else:
            adata.var["n_cells"] = number
        adata._inplace_subset_var(gene_subset)
        return adata if copy else None

    X = data  # proceed with processing the data matrix
    min_number = min_counts if min_cells is None else min_cells
    max_number = max_counts if max_cells is None else max_cells
    number_per_gene = axis_sum(
        X if min_cells is None and max_cells is None else X > 0, axis=0
    )
    if sparse.issparse(X):
        number_per_gene = number_per_gene.A1
    if min_number is not None:
        gene_subset = number_per_gene >= min_number
    if max_number is not None:
        gene_subset = number_per_gene <= max_number

    s = axis_sum(~gene_subset)
    if s > 0:
        msg = f"filtered out {s} genes that are detected "
        if min_cells is not None or min_counts is not None:
            msg += "in less than "
            msg += (
                f"{min_cells} cells" if min_counts is None else f"{min_counts} counts"
            )
        if max_cells is not None or max_counts is not None:
            msg += "in more than "
            msg += (
                f"{max_cells} cells" if max_counts is None else f"{max_counts} counts"
            )
        # logg.info(msg)
    return gene_subset, number_per_gene
