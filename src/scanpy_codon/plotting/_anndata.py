from .._utils import (
	_check_use_raw,
	# _doc_params,
	# _empty,
	# get_literal_vals,
	# sanitize_anndata,
)
from ._utils import (
	# ColorLike,
	_deprecated_scale,
	# _dk,
	# check_colornorm,
	# scatter_base,
	# scatter_group,
	# setup_axes,
)
from ..get import *


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
					f"The column `adata.obs[{groupby!r}]` needs to be categorical, "
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
