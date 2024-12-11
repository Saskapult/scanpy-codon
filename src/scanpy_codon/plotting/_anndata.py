from .._utils import (
	_check_use_raw,
	get_literal_vals,
)
from _utils import (
	_deprecated_scale,
	scatter_base,
	scatter_group,
	setup_axes,
)
from ..get import *
from python import collections.abc
import _utils
from python import numpy as np
from python import matplotlib 
from python import pandas as pd
from python import typing


# Run as python becuase it isn't likely to be doing much work
@python
def violin(
	adata,
	keys: List[str],
	groupby: str = None,
	log: bool = False,
	use_raw: bool = None,
	stripplot: bool = True,
	# jitter: float | bool = True,
	jitter: bool = True,
	size: int = 1,
	layer: str = None,
	density_norm: str = "width",
	order: List[str] = None,
	multi_panel: bool = None,
	xlabel: str = "",
	ylabel: List[str] = None,
	rotation: float = None,
	show: bool = None,
	# save: bool | str | None = None,
	save: str = None,
	ax = None,
	# deprecatd
	scale = None,
	**kwds,
):
	import seaborn as sns  # Slow import, only import if called

	# sanitize_anndata(adata)
	adata._sanitize()

	use_raw = _check_use_raw(adata, use_raw)
	if isinstance(keys, str):
		keys = [keys]
	keys = list(OrderedDict.fromkeys(keys))  # remove duplicates, preserving the order
	density_norm = _deprecated_scale(density_norm, scale, default="width")
	del scale

	if isinstance(ylabel, str | NoneType):
		ylabel = [ylabel] * (1 if groupby is None else len(keys))
	if groupby is None:
		if len(ylabel) != 1:
			raise ValueError(
				f"Expected number of y-labels to be `1`, found `{len(ylabel)}`."
			)
	elif len(ylabel) != len(keys):
		raise ValueError(
			f"Expected number of y-labels to be `{len(keys)}`, "
			f"found `{len(ylabel)}`."
		)

	if groupby is not None:
		obs_df = get.obs_df(adata, keys=[groupby] + keys, layer=layer, use_raw=use_raw)
		if kwds.get("palette") is None:
			if not isinstance(adata.obs[groupby].dtype, CategoricalDtype):
				raise ValueError(
					f"The column `adata.obs[{groupby}]` needs to be categorical, "
					f"but is of dtype {adata.obs[groupby].dtype}."
				)
			_utils.add_colors_for_categorical_sample_annotation(adata, groupby)
			kwds["hue"] = groupby
			kwds["palette"] = dict(
				zip(obs_df[groupby].cat.categories, adata.uns[f"{groupby}_colors"])
			)
	else:
		obs_df = get.obs_df(adata, keys=keys, layer=layer, use_raw=use_raw)
	if groupby is None:
		obs_tidy = pd.melt(obs_df, value_vars=keys)
		x = "variable"
		ys = ["value"]
	else:
		obs_tidy = obs_df
		x = groupby
		ys = keys

	if multi_panel and groupby is None and len(ys) == 1:
		# This is a quick and dirty way for adapting scales across several
		# keys if groupby is None.
		y = ys[0]

		g: sns.axisgrid.FacetGrid = sns.catplot(
			y=y,
			data=obs_tidy,
			kind="violin",
			density_norm=density_norm,
			col=x,
			col_order=keys,
			sharey=False,
			cut=0,
			inner=None,
			**kwds,
		)

		if stripplot:
			grouped_df = obs_tidy.groupby(x, observed=True)
			for ax_id, key in zip(range(g.axes.shape[1]), keys):
				sns.stripplot(
					y=y,
					data=grouped_df.get_group(key),
					jitter=jitter,
					size=size,
					color="black",
					ax=g.axes[0, ax_id],
				)
		if log:
			g.set(yscale="log")
		g.set_titles(col_template="{col_name}").set_xlabels("")
		if rotation is not None:
			for ax in g.axes[0]:
				ax.tick_params(axis="x", labelrotation=rotation)
	else:
		# set by default the violin plot cut=0 to limit the extend
		# of the violin plot (see stacked_violin code) for more info.
		kwds.setdefault("cut", 0)
		kwds.setdefault("inner")

		if ax is None:
			axs, _, _, _ = setup_axes(
				ax,
				panels=["x"] if groupby is None else keys,
				show_ticks=True,
				right_margin=0.3,
			)
		else:
			axs = [ax]
		for ax, y, ylab in zip(axs, ys, ylabel):
			ax = sns.violinplot(
				x=x,
				y=y,
				data=obs_tidy,
				order=order,
				orient="vertical",
				density_norm=density_norm,
				ax=ax,
				**kwds,
			)
			if stripplot:
				ax = sns.stripplot(
					x=x,
					y=y,
					data=obs_tidy,
					order=order,
					jitter=jitter,
					color="black",
					size=size,
					ax=ax,
				)
			if xlabel == "" and groupby is not None and rotation is None:
				xlabel = groupby.replace("_", " ")
			ax.set_xlabel(xlabel)
			if ylab is not None:
				ax.set_ylabel(ylab)

			if log:
				ax.set_yscale("log")
			if rotation is not None:
				ax.tick_params(axis="x", labelrotation=rotation)
	show = settings.autoshow if show is None else show
	_utils.savefig_or_show("violin", show=show, save=save)
	if show:
		return None
	if multi_panel and groupby is None and len(ys) == 1:
		return g
	if len(axs) == 1:
		return axs[0]
	return axs


def _scatter_obs(
    adata,
    x: str = None,
    y: str = None,
    color: List[str] = None,
    use_raw: bool = None,
    layers: List[str] = None,
    sort_order: bool = True,
    alpha: float = None,
    basis = None,
    groups: List[str] = None,
    components: List[str] = None,
    projection: str = "2d",
    legend_loc: str = "right margin",
    legend_fontsize: float = None,
    legend_fontweight: int = None,
    legend_fontoutline: float = None,
    color_map: str = None,
    palette = None,
    frameon: bool = None,
    right_margin: float = None,
    left_margin: float = None,
    size: float = None,
    marker: List[str] = ".",
    title: str = None,
    show: bool = None,
    save = None,
    ax = None,
):
    """See docstring of scatter."""
    # sanitize_anndata(adata)
    adata._sanitize()

    use_raw = _check_use_raw(adata, use_raw)

    # Process layers
    if layers in ["X", None] or (isinstance(layers, str) and layers in adata.layers):
        layers = (layers, layers, layers)
    elif isinstance(layers, abc.Collection) and len(layers) == 3:
        layers = tuple(layers)
        for layer in layers:
            if layer not in adata.layers and layer not in ["X", None]:
                raise ValueError(
                    "`layers` should have elements that are "
                    "either None or in adata.layers.keys()."
                )
    else:
        raise ValueError(
            "`layers` should be a string or a collection of strings "
            f"with length 3, had value '{layers}'"
        )
    if use_raw and layers not in [("X", "X", "X"), (None, None, None)]:
        ValueError("`use_raw` must be `False` if layers are used.")

    # valid_legend_locs = _utils._LegendLoc
    # if legend_loc not in valid_legend_locs:
    #     raise ValueError(
    #         f"Invalid `legend_loc`, need to be one of: {valid_legend_locs}."
    #     )
    if components is None:
        components = "1,2" if "2d" in projection else "1,2,3"
    if isinstance(components, str):
        components = components.split(",")
    components = np.array(components).astype(int) - 1
    keys = ["grey"] if color is None else color
    if title is not None and isinstance(title, str):
        title = [title]
    highlights = adata.uns.get("highlights", [])
    if basis is not None:
        try:
            # ignore the '0th' diffusion component
            if basis == "diffmap":
                components += 1
            Y = adata.obsm["X_" + basis][:, components]
            # correct the component vector for use in labeling etc.
            if basis == "diffmap":
                components -= 1
        except KeyError:
            raise KeyError(
                f"compute coordinates using visualization tool {basis} first"
            )
    elif x is not None and y is not None:
        if use_raw:
            if x in adata.obs.columns:
                x_arr = adata.obs_vector(x)
            else:
                x_arr = adata.raw.obs_vector(x)
            if y in adata.obs.columns:
                y_arr = adata.obs_vector(y)
            else:
                y_arr = adata.raw.obs_vector(y)
        else:
            x_arr = adata.obs_vector(x, layer=layers[0])
            y_arr = adata.obs_vector(y, layer=layers[1])

        Y = np.c_[x_arr, y_arr]
    else:
        raise ValueError("Either provide a `basis` or `x` and `y`.")

    if size is None:
        n = Y.shape[0]
        size = 120000 / n

    if legend_fontsize is None:
        legend_fontsize = matplotlib.rcParams["legend.fontsize"]

    palette_was_none = False
    if palette is None:
        palette_was_none = True
    if isinstance(palette, abc.Sequence) and not isinstance(palette, str):
        palettes = palette if not matplotlib.colors.is_color_like(palette[0]) else [palette]
    else:
        palettes = [palette for _ in range(len(keys))]
    palettes = [_utils.default_palette(palette) for palette in palettes]

    if basis is not None:
        component_name = (
            "DC"
            if basis == "diffmap"
            else "tSNE"
            if basis == "tsne"
            else "UMAP"
            if basis == "umap"
            else "PC"
            if basis == "pca"
            else "TriMap"
            if basis == "trimap"
            else basis.replace("draw_graph_", "").upper()
            if "draw_graph" in basis
            else basis
        )
    else:
        component_name = None
    axis_labels = (x, y) if component_name is None else None
    show_ticks = component_name is None

    # generate the colors
    color_ids = []
    categoricals = []
    colorbars = []
    for ikey, key in enumerate(keys):
        c = "white"
        categorical = False  # by default, assume continuous or flat color
        colorbar = None
        # test whether we have categorial or continuous annotation
        if key in adata.obs_keys():
            if isinstance(adata.obs[key].dtype, pd.api.types.CategoricalDtype):
                categorical = True
            else:
                c = adata.obs[key].to_numpy()
        # coloring according to gene expression
        elif use_raw and adata.raw is not None and key in adata.raw.var_names:
            c = adata.raw.obs_vector(key)
        elif key in adata.var_names:
            c = adata.obs_vector(key, layer=layers[2])
        elif matplotlib.colors.is_color_like(key):  # a flat color
            c = key
            colorbar = False
        else:
            raise ValueError(
                f"key {key} is invalid! pass valid observation annotation, "
                f"one of {adata.obs_keys()} or a gene name {adata.var_names}"
            )
        if colorbar is None:
            colorbar = not categorical
        colorbars.append(colorbar)
        if categorical:
            categoricals.append(ikey)
        color_ids.append(c)

    if right_margin is None and len(categoricals) > 0 and legend_loc == "right margin":
        right_margin = 0.5
    if title is None and keys[0] is not None:
        title = [
            key.replace("_", " ") if not matplotlib.colors.is_color_like(key) else "" for key in keys
        ]

    axs = scatter_base(
        Y,
        title=title,
        alpha=alpha,
        component_name=component_name,
        axis_labels=axis_labels,
        component_indexnames=components + 1,
        projection=projection,
        colors=color_ids,
        highlights=highlights,
        colorbars=colorbars,
        right_margin=right_margin,
        left_margin=left_margin,
        sizes=[size for _ in keys],
        markers=marker,
        color_map=color_map,
        show_ticks=show_ticks,
        ax=ax,
    )

    def add_centroid(centroids, name, Y, mask):
        Y_mask = Y[mask]
        if Y_mask.shape[0] == 0:
            return
        median = np.median(Y_mask, axis=0)
        i = np.argmin(np.sum(np.abs(Y_mask - median), axis=1))
        centroids[name] = Y_mask[i]

    # loop over all categorical annotation and plot it
    for ikey, palette in zip(categoricals, palettes):
        key = keys[ikey]
        _utils.add_colors_for_categorical_sample_annotation(
            adata, key, palette=palette, force_update_colors=not palette_was_none
        )
        # actually plot the groups
        mask_remaining = np.ones(Y.shape[0], dtype=bool)
        centroids = {}
        if groups is None:
            for iname, name in enumerate(adata.obs[key].cat.categories):
                # if name not in settings.categories_to_ignore:
                mask = scatter_group(
                    axs[ikey],
                    key,
                    iname,
                    adata,
                    Y,
                    projection=projection,
                    size=size,
                    alpha=alpha,
                    marker=marker,
                )
                mask_remaining[mask] = False
                if legend_loc.startswith("on data"):
                    add_centroid(centroids, name, Y, mask)
        else:
            groups = [groups] if isinstance(groups, str) else groups
            for name in groups:
                if name not in set(adata.obs[key].cat.categories):
                    raise ValueError(
                        f"{name} is invalid! specify valid name, "
                        f"one of {adata.obs[key].cat.categories}"
                    )
                else:
                    iname = np.flatnonzero(
                        adata.obs[key].cat.categories.values == name
                    )[0]
                    mask = scatter_group(
                        axs[ikey],
                        key,
                        iname,
                        adata,
                        Y,
                        projection=projection,
                        size=size,
                        alpha=alpha,
                        marker=marker,
                    )
                    if legend_loc.startswith("on data"):
                        add_centroid(centroids, name, Y, mask)
                    mask_remaining[mask] = False
        if mask_remaining.sum() > 0:
            data = [Y[mask_remaining, 0], Y[mask_remaining, 1]]
            if projection == "3d":
                data.append(Y[mask_remaining, 2])
            axs[ikey].scatter(
                *data,
                marker=marker,
                c="lightgrey",
                s=size,
                edgecolors="none",
                zorder=-1,
            )
        legend = None
        if legend_loc.startswith("on data"):
            if legend_fontweight is None:
                legend_fontweight = "bold"
            if legend_fontoutline is not None:
                path_effect = [
                    matplotlib.patheffects.withStroke(linewidth=legend_fontoutline, foreground="w")
                ]
            else:
                path_effect = None
            for name, pos in centroids.items():
                axs[ikey].text(
                    pos[0],
                    pos[1],
                    name,
                    weight=legend_fontweight,
                    verticalalignment="center",
                    horizontalalignment="center",
                    fontsize=legend_fontsize,
                    path_effects=path_effect,
                )

            all_pos = np.zeros((len(adata.obs[key].cat.categories), 2))
            for iname, name in enumerate(adata.obs[key].cat.categories):
                all_pos[iname] = centroids.get(name, [np.nan, np.nan])
            # if legend_loc == "on data export":
            #     filename = settings.writedir / "pos.csv"
            #     logg.warning(f"exporting label positions to {filename}")
            #     settings.writedir.mkdir(parents=True, exist_ok=True)
            #     np.savetxt(filename, all_pos, delimiter=",")
        elif legend_loc == "right margin":
            legend = axs[ikey].legend(
                frameon=False,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                ncol=(
                    1
                    if len(adata.obs[key].cat.categories) <= 14
                    else 2
                    if len(adata.obs[key].cat.categories) <= 30
                    else 3
                ),
                fontsize=legend_fontsize,
            )
        elif legend_loc != "none":
            legend = axs[ikey].legend(
                frameon=False, loc=legend_loc, fontsize=legend_fontsize
            )
        if legend is not None:
            _attr = "legend_handles"
            for handle in getattr(legend, _attr):
                handle.set_sizes([300.0])

    # # draw a frame around the scatter
    # frameon = settings._frameon if frameon is None else frameon
    # if not frameon and x is None and y is None:
    #     for ax in axs:
    #         ax.set_xlabel("")
    #         ax.set_ylabel("")
    #         ax.set_frame_on(False)

    show = True # settings.autoshow if show is None else show
    _utils.savefig_or_show("scatter" if basis is None else basis, show=show, save=save)
    if show:
        return None
    if len(keys) > 1:
        return axs
    return axs[0]



def _check_if_annotations(
    adata,
    axis_name,
    x: str = None,
    y: str = None,
    colors = None,
    use_raw: bool = None,
) -> bool:
    """Checks if `x`, `y`, and `colors` are annotations of `adata`.
    In the case of `colors`, valid matplotlib colors are also accepted.

    If `axis_name` is `obs`, checks in `adata.obs.columns` and `adata.var_names`,
    if `axis_name` is `var`, checks in `adata.var.columns` and `adata.obs_names`.
    """
    annotations = getattr(adata, axis_name).columns
    other_ax_obj = (
        adata.raw if _check_use_raw(adata, use_raw) and axis_name == "obs" else adata
    )
    names = getattr(
        other_ax_obj, "var" if axis_name == "obs" else "obs"
    ).index

    def is_annotation(needle):
        return needle.isin({None}) | needle.isin(annotations) | needle.isin(names)

    if not is_annotation(pd.Index([x, y])).all():
        return False

    color_idx = pd.Index(colors if colors is not None else [])
    # Colors are valid
    color_valid = np.fromiter(
        map(matplotlib.colors.is_color_like, color_idx), dtype=np.bool_, count=len(color_idx)
    )
    # Annotation names are valid too
    color_valid[~color_valid] = is_annotation(color_idx[~color_valid])
    return bool(color_valid.all())


def scatter(
    adata,
    x: str = None,
    y: str = None,
    # color: str | ColorLike | Collection[str | ColorLike] | None = None,
    color: str = None,
    use_raw: bool = None,
    layers: List[str] = None,
    sort_order: bool = True,
    alpha: float = None,
    basis = None,
    groups: List[str] = None,
    components: List[str] = None,
    projection: str = "2d",
    legend_loc: str = "right margin",
    legend_fontsize: float = None,
    legend_fontweight: int = None,
    legend_fontoutline: float = None,
    color_map: str = None,
    # color_map: str | Colormap | None = None,
    # palette: Cycler | ListedColormap | ColorLike | Sequence[ColorLike] | None = None,
    palette = None,
    frameon: bool = None,
    right_margin: float = None,
    left_margin: float = None,
    size: float = None,
    marker: str = ".",
    title: str = None,
    show: bool = None,
    save = None,
    ax = None,
):
    # color can be a obs column name or a matplotlib color specification (or a collection thereof)
    if color is not None:
        color = typing.cast(
            abc.Collection[str],
            [color] if isinstance(color, str) or matplotlib.colors.is_color_like(color) else color,
        )
    args = {}#locals()

    if basis is not None:
        return _scatter_obs(**args)
    if x is None or y is None:
        raise ValueError("Either provide a `basis` or `x` and `y`.")
    if _check_if_annotations(adata, "obs", x=x, y=y, colors=color, use_raw=use_raw):
        return _scatter_obs(**args)
    if _check_if_annotations(adata, "var", x=x, y=y, colors=color, use_raw=use_raw):
        args_t = {**args, "adata": adata.T}
        axs = _scatter_obs(**args_t)
        # store .uns annotations that were added to the new adata object
        adata.uns = args_t["adata"].uns
        return axs
    raise ValueError(
        "`x`, `y`, and potential `color` inputs must all "
        "come from either `.obs` or `.var`"
    )
