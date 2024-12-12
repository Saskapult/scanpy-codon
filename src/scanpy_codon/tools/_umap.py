from .._utils import NeighborsView
from ._utils import get_init_pos_from_paga, _choose_representation
from python import numpy as np 
from python import sklearn.utils
from python import warnings


def umap(
    adata,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: int = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos = "spectral",
    random_state = 0,
    a: float = None,
    b: float = None,
    method = "umap",
    key_added: str = None,
    neighbors_key: str = "neighbors",
    copy: bool = False,
):
    adata = adata.copy() if copy else adata

    key_obsm, key_uns = ("X_umap", "umap") if key_added is None else [key_added] * 2

    if neighbors_key is None:  # backwards compat
        neighbors_key = "neighbors"
    if neighbors_key not in adata.uns:
        raise ValueError(
            f"Did not find .uns[{neighbors_key}]. Run `sc.pp.neighbors` first."
        )

    # start = logg.info("computing UMAP")

    neighbors = NeighborsView(adata, neighbors_key)

    # if "params" not in neighbors or neighbors["params"]["method"] != "umap":
    #     logg.warning(
    #         f'.obsp["{neighbors["connectivities_key"]}"] have not been computed using umap'
    #     )

    # with warnings.catch_warnings():
    #     # umap 0.5.0
    #     warnings.filterwarnings("ignore", message=r"Tensorflow not installed")
    #     import umap
    from python import umap

    # from umap.umap_ import find_ab_params, simplicial_set_embedding

    if a is None or b is None:
        a, b = umap.umap_.find_ab_params(spread, min_dist)
    adata.uns[key_uns] = dict(params=dict(a=a, b=b))
    if isinstance(init_pos, str) and init_pos in adata.obsm:
        init_coords = adata.obsm[init_pos]
    elif isinstance(init_pos, str) and init_pos == "paga":
        init_coords = get_init_pos_from_paga(
            adata, random_state=random_state, neighbors_key=neighbors_key
        )
    else:
        init_coords = init_pos  # Let umap handle it
    if hasattr(init_coords, "dtype"):
        init_coords = utils.check_array(init_coords, dtype=np.float32, accept_sparse=False)

    if random_state != 0:
        adata.uns[key_uns]["params"]["random_state"] = random_state
    random_state = utils.check_random_state(random_state)

    neigh_params = neighbors["params"]
    X = _choose_representation(
        adata,
        use_rep=neigh_params.get("use_rep", None),
        n_pcs=neigh_params.get("n_pcs", None),
        silent=True,
    )
    if method == "umap":
        # the data matrix X is really only used for determining the number of connected components
        # for the init condition in the UMAP embedding
        default_epochs = 500 if neighbors["connectivities"].shape[0] <= 10000 else 200
        n_epochs = default_epochs if maxiter is None else maxiter
        X_umap, _ = umap.umap_.simplicial_set_embedding(
            data=X,
            graph=neighbors["connectivities"].tocoo(),
            n_components=n_components,
            initial_alpha=alpha,
            a=a,
            b=b,
            gamma=gamma,
            negative_sample_rate=negative_sample_rate,
            n_epochs=n_epochs,
            init=init_coords,
            random_state=random_state,
            metric=neigh_params.get("metric", "euclidean"),
            metric_kwds=neigh_params.get("metric_kwds", {}),
            densmap=False,
            densmap_kwds={},
            output_dens=False,
            verbose=1 > 3,
        )
    elif method == "rapids":
        msg = (
            "`method='rapids'` is deprecated. "
            "Use `rapids_singlecell.tl.louvain` instead."
        )
        warnings.warn(msg)
        metric = neigh_params.get("metric", "euclidean")
        if metric != "euclidean":
            raise ValueError(
                f"`sc.pp.neighbors` was called with `metric` {metric}, "
                "but umap `method` 'rapids' only supports the 'euclidean' metric."
            )
        from python import cuml

        n_neighbors = neighbors["params"]["n_neighbors"]
        n_epochs = (
            500 if maxiter is None else maxiter
        )  # 0 is not a valid value for rapids, unlike original umap
        X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
        umap = cuml.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            n_epochs=n_epochs,
            learning_rate=alpha,
            init=init_pos,
            min_dist=min_dist,
            spread=spread,
            negative_sample_rate=negative_sample_rate,
            a=a,
            b=b,
            verbose=1 > 3,
            random_state=random_state,
        )
        X_umap = umap.fit_transform(X_contiguous)
    adata.obsm[key_obsm] = X_umap  # annotate samples with UMAP coordinates
    # logg.info(
    #     "    finished",
    #     time=start,
    #     deep=(
    #         "added\n"
    #         f"    {key_obsm!r}, UMAP coordinates (adata.obsm)\n"
    #         f"    {key_uns!r}, UMAP parameters (adata.uns)"
    #     ),
    # )
    return adata if copy else None
