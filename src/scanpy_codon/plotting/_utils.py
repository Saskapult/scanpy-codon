from python import warnings
from python import numpy as np
from python import matplotlib
from python import matplotlib.pyplot as plt
from python import matplotlib.figure
from python import collections.abc
from python import cycler 
import palettes
# import ..logging as logg
# from .. import logging as logg
# import ../logging as logg
# import scanpy_codon.logging as logg
# import .logging as logg
# import logging as logg
# from scanpy_codon import logging as logg


_LegendLoc = [
    "none",
    "right margin",
    "on data",
    "on data export",
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]

additional_colors = {
    "gold2": "#eec900",
    "firebrick3": "#cd2626",
    "khaki2": "#eee685",
    "slategray3": "#9fb6cd",
    "palegreen3": "#7ccd7c",
    "tomato2": "#ee5c42",
    "grey80": "#cccccc",
    "grey90": "#e5e5e5",
    "wheat4": "#8b7e66",
    "grey65": "#a6a6a6",
    "grey10": "#1a1a1a",
    "grey20": "#333333",
    "grey50": "#7f7f7f",
    "grey30": "#4d4d4d",
    "grey40": "#666666",
    "antiquewhite2": "#eedfcc",
    "grey77": "#c4c4c4",
    "snow4": "#8b8989",
    "chartreuse3": "#66cd00",
    "yellow4": "#8b8b00",
    "darkolivegreen2": "#bcee68",
    "olivedrab3": "#9acd32",
    "azure3": "#c1cdcd",
    "violetred": "#d02090",
    "mediumpurple3": "#8968cd",
    "purple4": "#551a8b",
    "seagreen4": "#2e8b57",
    "lightblue3": "#9ac0cd",
    "orchid3": "#b452cd",
    "indianred 3": "#cd5555",
    "grey60": "#999999",
    "mediumorchid1": "#e066ff",
    "plum3": "#cd96cd",
    "palevioletred3": "#cd6889",
}


def _deprecated_scale(
    density_norm,
    scale,
    # default: DensityNorm | Empty = _empty,
):
    # if scale is _empty:
    if scale is None:
        return density_norm
    # if density_norm != default:
    #     msg = "can’t specify both `scale` and `density_norm`"
    #     raise ValueError(msg)
    msg = "`scale` is deprecated, use `density_norm` instead"
    warnings.warn(msg)
    return scale


def ticks_formatter(x, pos):
    # pretty scientific notation
    if False:
        a, b = f"{x:.2e}".split("e")
        b = int(b)
        return rf"${a} \times 10^{{{b}}}$"
    else:
        return f"{x:.3f}".rstrip("0").rstrip(".")


def check_projection(projection):
    """Validation for projection argument."""
    if projection not in {"2d", "3d"}:
        raise ValueError(f"Projection must be '2d' or '3d', was '{projection}'.")
    # if projection == "3d":
    #     from packaging.version import parse

    #     mpl_version = parse(mpl.__version__)
    #     if mpl_version < parse("3.3.3"):
    #         raise ImportError(
    #             f"3d plotting requires matplotlib > 3.3.3. Found {mpl.__version__}"
    #         )


def setup_axes(
    ax = None,
    panels="blue",
    colorbars=(False,),
    right_margin=None,
    left_margin=None,
    projection: str = "2d",
    show_ticks=False,
):
    """Grid of axes for plotting, legends and colorbars."""
    check_projection(projection)
    if left_margin is not None:
        raise NotImplementedError("We currently don’t support `left_margin`.")
    if np.any(colorbars) and right_margin is None:
        right_margin = 1 - matplotlib.rcParams["figure.subplot.right"] + 0.21  # 0.25
    elif right_margin is None:
        right_margin = 1 - matplotlib.rcParams["figure.subplot.right"] + 0.06  # 0.10
    # make a list of right margins for each panel
    if not isinstance(right_margin, list):
        right_margin_list = [right_margin for i in range(len(panels))]
    else:
        right_margin_list = right_margin

    # make a figure with len(panels) panels in a row side by side
    top_offset = 1 - matplotlib.rcParams["figure.subplot.top"]
    bottom_offset = 0.15 if show_ticks else 0.08
    left_offset = 1 if show_ticks else 0.3  # in units of base_height
    base_height = matplotlib.rcParams["figure.figsize"][1]
    height = base_height
    base_width = matplotlib.rcParams["figure.figsize"][0]
    if show_ticks:
        base_width *= 1.1

    draw_region_width = (
        base_width - left_offset - top_offset - 0.5
    )  # this is kept constant throughout

    right_margin_factor = sum([1 + right_margin for right_margin in right_margin_list])
    width_without_offsets = (
        right_margin_factor * draw_region_width
    )  # this is the total width that keeps draw_region_width

    right_offset = (len(panels) - 1) * left_offset
    figure_width = width_without_offsets + left_offset + right_offset
    draw_region_width_frac = draw_region_width / figure_width
    left_offset_frac = left_offset / figure_width
    right_offset_frac = (  # noqa: F841  # TODO Does this need fixing?
        1 - (len(panels) - 1) * left_offset_frac
    )

    if ax is None:
        plt.figure(
            figsize=(figure_width, height),
            subplotpars=matplotlib.figure.SubplotParams(left=0, right=1, bottom=bottom_offset),
        )
    left_positions = [left_offset_frac, left_offset_frac + draw_region_width_frac]
    for i in range(1, len(panels)):
        right_margin = right_margin_list[i - 1]
        left_positions.append(
            left_positions[-1] + right_margin * draw_region_width_frac
        )
        left_positions.append(left_positions[-1] + draw_region_width_frac)
    panel_pos = [[bottom_offset], [1 - top_offset], left_positions]

    axs = []
    if ax is None:
        for icolor, color in enumerate(panels):
            left = panel_pos[2][2 * icolor]
            bottom = panel_pos[0][0]
            width = draw_region_width / figure_width
            height = panel_pos[1][0] - bottom
            if projection == "2d":
                ax = plt.axes([left, bottom, width, height])
            elif projection == "3d":
                ax = plt.axes([left, bottom, width, height], projection="3d")
            axs.append(ax)
    else:
        axs = ax if isinstance(ax, abc.Sequence) else [ax]

    return axs, panel_pos, draw_region_width, figure_width


def scatter_base(
    Y,
    # colors: str | Sequence[ColorLike | np.ndarray] = "blue",
    colors: str = "blue",
    sort_order=True,
    alpha=None,
    highlights=(),
    right_margin=None,
    left_margin=None,
    projection: str = "2d",
    title=None,
    component_name="DC",
    component_indexnames=(1, 2, 3),
    axis_labels=None,
    colorbars=(False,),
    sizes=(1,),
    markers=".",
    color_map="viridis",
    show_ticks=True,
    ax=None,
):
    if isinstance(highlights, abc.Mapping):
        highlights_indices = sorted(highlights)
        highlights_labels = [highlights[i] for i in highlights_indices]
    else:
        highlights_indices = highlights
        highlights_labels = []
    # if we have a single array, transform it into a list with a single array
    if isinstance(colors, str):
        colors = [colors]
    if isinstance(markers, str):
        markers = [markers]
    if len(sizes) != len(colors) and len(sizes) == 1:
        sizes = [sizes[0] for _ in range(len(colors))]
    if len(markers) != len(colors) and len(markers) == 1:
        markers = [markers[0] for _ in range(len(colors))]
    axs, panel_pos, draw_region_width, figure_width = setup_axes(
        ax,
        panels=colors,
        colorbars=colorbars,
        projection=projection,
        right_margin=right_margin,
        left_margin=left_margin,
        show_ticks=show_ticks,
    )
    for icolor, color in enumerate(colors):
        ax = axs[icolor]
        marker = markers[icolor]
        bottom = panel_pos[0][0]
        height = panel_pos[1][0] - bottom
        Y_sort = Y
        if not matplotlib.colors.is_color_like(color) and sort_order:
            sort = np.argsort(color)
            color = color[sort]
            Y_sort = Y[sort]
        if projection == "2d":
            data = Y_sort[:, 0], Y_sort[:, 1]
        elif projection == "3d":
            data = Y_sort[:, 0], Y_sort[:, 1], Y_sort[:, 2]
        else:
            # raise ValueError(f"Unknown projection {projection!r} not in '2d', '3d'")
            raise ValueError(f"Unknown projection {projection} not in '2d', '3d'")
        if not isinstance(color, str) or color != "white":
            sct = ax.scatter(
                *data,
                marker=marker,
                c=color,
                alpha=alpha,
                edgecolors="none",  # 'face',
                s=sizes[icolor],
                cmap=color_map,
                rasterized=False,
                # rasterized=settings._vector_friendly,
            )
        if colorbars[icolor]:
            width = 0.006 * draw_region_width / len(colors)
            left = (
                panel_pos[2][2 * icolor + 1]
                + (1.2 if projection == "3d" else 0.2) * width
            )
            rectangle = [left, bottom, width, height]
            fig = plt.gcf()
            ax_cb = fig.add_axes(rectangle)
            _ = plt.colorbar(
                sct, format=matplotlib.ticker.FuncFormatter(ticks_formatter), cax=ax_cb
            )
        # set the title
        if title is not None:
            ax.set_title(title[icolor])
        # output highlighted data points
        for iihighlight, ihighlight in enumerate(highlights_indices):
            ihighlight = ihighlight if isinstance(ihighlight, int) else int(ihighlight)
            data = [Y[ihighlight, 0]], [Y[ihighlight, 1]]
            if "3d" in projection:
                data = [Y[ihighlight, 0]], [Y[ihighlight, 1]], [Y[ihighlight, 2]]
            ax.scatter(
                *data,
                c="black",
                facecolors="black",
                edgecolors="black",
                marker="x",
                s=10,
                zorder=20,
            )
            highlight_text = (
                highlights_labels[iihighlight]
                if len(highlights_labels) > 0
                else str(ihighlight)
            )
            # the following is a Python 2 compatibility hack
            ax.text(
                *([d[0] for d in data] + [highlight_text]),
                zorder=20,
                fontsize=10,
                color="black",
            )
        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
            if "3d" in projection:
                ax.set_zticks([])
    # set default axis_labels
    if axis_labels is None:
        axis_labels = [
            [component_name + str(i) for i in component_indexnames]
            for _ in range(len(axs))
        ]
    else:
        axis_labels = [axis_labels for _ in range(len(axs))]
    for iax, ax in enumerate(axs):
        ax.set_xlabel(axis_labels[iax][0])
        ax.set_ylabel(axis_labels[iax][1])
        if "3d" in projection:
            # shift the label closer to the axis
            ax.set_zlabel(axis_labels[iax][2], labelpad=-7)
    for ax in axs:
        # scale limits to match data
        ax.autoscale_view()
    return axs


def scatter_group(
    ax,
    key: str,
    cat_code: int,
    adata,
    Y,
    projection: str = "2d",
    size: int = 3,
    alpha: float = None,
    marker = ".",
):
    """Scatter of group using representation of data Y."""
    mask_obs = adata.obs[key].cat.categories[cat_code] == adata.obs[key].values
    color = adata.uns[key + "_colors"][cat_code]
    if not isinstance(color[0], str):
        from python import matplotlib.colors

        color = colors.rgb2hex(adata.uns[key + "_colors"][cat_code])
    if not matplotlib.colors.is_color_like(color):
        raise ValueError(f'"{color}" is not a valid matplotlib color.')
    data = [Y[mask_obs, 0], Y[mask_obs, 1]]
    if projection == "3d":
        data.append(Y[mask_obs, 2])
    ax.scatter(
        *data,
        marker=marker,
        alpha=alpha,
        c=color,
        edgecolors="none",
        s=size,
        label=adata.obs[key].cat.categories[cat_code],
        rasterized=False,
    )
    return mask_obs


def default_palette(
    palette = None,
):
    if palette is None:
        return matplotlib.rcParams["axes.prop_cycle"]
    elif not (isinstance(palette, cycler.Cycler) or isinstance(palette, str)):
        return cycler.cycler(color=palette)
    else:
        # return palettes
        print('Unimplemented bit')


def _set_colors_for_categorical_obs(
    adata, value_to_plot, palette
):
    # if adata.obs[value_to_plot].dtype == bool:
    #     categories = (
    #         adata.obs[value_to_plot].astype(str).astype("category").cat.categories
    #     )
    # else:
    #     categories = adata.obs[value_to_plot].cat.categories
    categories = adata.obs[value_to_plot].cat.categories
    # check is palette is a valid matplotlib colormap
    if isinstance(palette, str) and palette in plt.colormaps():
        # this creates a palette from a colormap. E.g. 'Accent, Dark2, tab20'
        cmap = plt.get_cmap(palette)
        colors_list = [matplotlib.colors.to_hex(x) for x in cmap(np.linspace(0, 1, len(categories)))]
    elif isinstance(palette, abc.Mapping):
        colors_list = [matplotlib.colors.to_hex(palette[k], keep_alpha=True) for k in categories]
    else:
        # check if palette is a list and convert it to a cycler, thus
        # it doesnt matter if the list is shorter than the categories length:
        if isinstance(palette, abc.Sequence):
            # if len(palette) < len(categories):
            #     logg.warning(
            #         "Length of palette colors is smaller than the number of "
            #         f"categories (palette length: {len(palette)}, "
            #         f"categories length: {len(categories)}. "
            #         "Some categories will have the same color."
            #     )
            # check that colors are valid
            _color_list = []
            for color in palette:
                if not matplotlib.colors.is_color_like(color):
                    # check if the color is a valid R color and translate it
                    # to a valid hex color value
                    if color in additional_colors:
                        color = additional_colors[color]
                    else:
                        raise ValueError(
                            "The following color value of the given palette "
                            f"is not valid: {color}"
                        )
                _color_list.append(color)

            palette = cycler.cycler(color=_color_list)
        if not isinstance(palette, cycler.Cycler):
            raise ValueError(
                "Please check that the value of 'palette' is a valid "
                "matplotlib colormap string (eg. Set2), a  list of color names "
                "or a cycler with a 'color' key."
            )
        if "color" not in palette.keys:
            raise ValueError("Please set the palette key 'color'.")

        cc = palette()
        colors_list = [matplotlib.colors.to_hex(next(cc)["color"]) for x in range(len(categories))]

    adata.uns[value_to_plot + "_colors"] = colors_list


def _set_default_colors_for_categorical_obs(adata, value_to_plot):
    # if adata.obs[value_to_plot].dtype == bool:
    #     categories = (
    #         adata.obs[value_to_plot].astype(str).astype("category").cat.categories
    #     )
    # else:
    #     categories = adata.obs[value_to_plot].cat.categories
    categories = adata.obs[value_to_plot].cat.categories

    length = len(categories)

    # check if default matplotlib palette has enough colors
    if len(matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]) >= length:
        cc = matplotlib.rcParams["axes.prop_cycle"]()
        palette = [next(cc)["color"] for _ in range(length)]

    else:
        if length <= 20:
            palette = palettes.default_20
        elif length <= 28:
            palette = palettes.default_28
        elif length <= len(palettes.default_102):  # 103 colors
            palette = palettes.default_102
        else:
            palette = ["grey" for _ in range(length)]
            # logg.info(
            #     f"the obs value {value_to_plot!r} has more than 103 categories. Uniform "
            #     "'grey' color will be used for all categories."
            # )

    _set_colors_for_categorical_obs(adata, value_to_plot, palette[:length])


def _validate_palette(adata, key: str) -> None:
    _palette = []
    color_key = f"{key}_colors"

    for color in adata.uns[color_key]:
        if not matplotlib.colors.is_color_like(color):
            # check if the color is a valid R color and translate it
            # to a valid hex color value
            if color in additional_colors:
                color = additional_colors[color]
            else:
                # logg.warning(
                #     f"The following color value found in adata.uns['{key}_colors'] "
                #     f"is not valid: '{color}'. Default colors will be used instead."
                # )
                _set_default_colors_for_categorical_obs(adata, key)
                _palette = None
                break
        _palette.append(color)
    # Don’t modify if nothing changed
    if _palette is None or np.array_equal(_palette, adata.uns[color_key]):
        return
    adata.uns[color_key] = _palette


def _set_default_colors_for_categorical_obs(adata, value_to_plot):
    # if adata.obs[value_to_plot].dtype == bool:
    #     categories = (
    #         adata.obs[value_to_plot].astype(str).astype("category").cat.categories
    #     )
    # else:
    #     categories = adata.obs[value_to_plot].cat.categories
    categories = adata.obs[value_to_plot].cat.categories

    length = len(categories)

    # check if default matplotlib palette has enough colors
    if len(matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]) >= length:
        cc = matplotlib.rcParams["axes.prop_cycle"]()
        palette = [next(cc)["color"] for _ in range(length)]

    else:
        if length <= 20:
            palette = palettes.default_20
        elif length <= 28:
            palette = palettes.default_28
        elif length <= len(palettes.default_102):  # 103 colors
            palette = palettes.default_102
        else:
            palette = ["grey" for _ in range(length)]
            # logg.info(
            #     f"the obs value {value_to_plot!r} has more than 103 categories. Uniform "
            #     "'grey' color will be used for all categories."
            # )

    _set_colors_for_categorical_obs(adata, value_to_plot, palette[:length])


def add_colors_for_categorical_sample_annotation(
    adata, key, palette=None, force_update_colors=False
):
    color_key = f"{key}_colors"
    colors_needed = len(adata.obs[key].cat.categories)
    if palette and force_update_colors:
        _set_colors_for_categorical_obs(adata, key, palette)
    elif color_key in adata.uns and len(adata.uns[color_key]) <= colors_needed:
        _validate_palette(adata, key)
    else:
        _set_default_colors_for_categorical_obs(adata, key)


def savefig_or_show(
    writekey: str,
    show: bool = None,
    dpi: int = None,
    ext: str = None,
    save = None,
):
    if isinstance(save, str):
        # check whether `save` contains a figure extension
        if ext is None:
            for try_ext in [".svg", ".pdf", ".png"]:
                if save.endswith(try_ext):
                    ext = try_ext[1:]
                    save = save.replace(try_ext, "")
                    break
        # append it
        writekey += save
        save = True
    save = False #settings.autosave if save is None else save
    show = True #settings.autoshow if show is None else show
    if save:
        # savefig(writekey, dpi=dpi, ext=ext)
        print('Implement saving pls')
    if show:
        plt.show()
    if save:
        plt.close()  # clear figure
