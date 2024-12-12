from python import numpy as np
from .._utils import _choose_graph
from python import warnings 


def get_init_pos_from_paga(
    adata, adjacency=None, random_state=0, neighbors_key=None, obsp=None
):
    np.random.seed(random_state)
    if adjacency is None:
        adjacency = _choose_graph(adata, obsp, neighbors_key)
    if "paga" in adata.uns and "pos" in adata.uns["paga"]:
        groups = adata.obs[adata.uns["paga"]["groups"]]
        pos = adata.uns["paga"]["pos"]
        connectivities_coarse = adata.uns["paga"]["connectivities"]
        init_pos = np.ones((adjacency.shape[0], 2))
        for i, group_pos in enumerate(pos):
            subset = (groups == groups.cat.categories[i]).values
            neighbors = connectivities_coarse[i].nonzero()
            if len(neighbors[1]) > 0:
                connectivities = connectivities_coarse[i][neighbors]
                nearest_neighbor = neighbors[1][np.argmax(connectivities)]
                noise = np.random.random((len(subset[subset]), 2))
                dist = pos[i] - pos[nearest_neighbor]
                noise = noise * dist
                init_pos[subset] = group_pos - 0.5 * dist + noise
            else:
                init_pos[subset] = group_pos
    else:
        raise ValueError("Plot PAGA first, so that adata.uns['paga']" "with key 'pos'.")
    return init_pos


def _choose_representation(
    adata,
    use_rep: str = None,
    n_pcs: int = None,
    silent: bool = False,
):  # TODO: what else?
    from ..preprocessing import pca

    verbosity = 1
    if silent and verbosity > 1:
        verbosity = 1
    if use_rep is None and n_pcs == 0:  # backwards compat for specifying `.X`
        use_rep = "X"
    if use_rep is None:
        if adata.n_vars > 50:
            if "X_pca" in adata.obsm:
                if n_pcs is not None and n_pcs > adata.obsm["X_pca"].shape[1]:
                    raise ValueError(
                        "`X_pca` does not have enough PCs. Rerun `sc.pp.pca` with adjusted `n_comps`."
                    )
                X = adata.obsm["X_pca"][:, :n_pcs]
                # logg.info(f"    using 'X_pca' with n_pcs = {X.shape[1]}")
            else:
                warnings.warn(
                    f"You’re trying to run this on {adata.n_vars} dimensions of `.X`, "
                    "if you really want this, set `use_rep='X'`.\n         "
                    "Falling back to preprocessing with `sc.pp.pca` and default params."
                )
                n_pcs_pca = n_pcs if n_pcs is not None else 50
                pca(adata, n_comps=n_pcs_pca)
                X = adata.obsm["X_pca"]
        else:
            # logg.info("    using data matrix X directly")
            X = adata.X
    else:
        if use_rep in adata.obsm and n_pcs is not None:
            if n_pcs > adata.obsm[use_rep].shape[1]:
                raise ValueError(
                    f"{use_rep} does not have enough Dimensions. Provide a "
                    "Representation with equal or more dimensions than"
                    "`n_pcs` or lower `n_pcs` "
                )
            X = adata.obsm[use_rep][:, :n_pcs]
        elif use_rep in adata.obsm and n_pcs is None:
            X = adata.obsm[use_rep]
        elif use_rep == "X":
            X = adata.X
        else:
            raise ValueError(
                f"Did not find {use_rep} in `.obsm.keys()`. "
                "You need to compute it first."
            )
    return X
