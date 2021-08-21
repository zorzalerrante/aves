import matplotlib.colors as colors
import numpy as np
import pandas as pd
import seaborn as sns
from mapclassify import FisherJenks, Quantiles

from aves.visualization.colors import MidpointNormalize, add_ranged_color_legend


def choropleth_map(
    ax,
    geodf,
    column,
    k=10,
    palette=None,
    default_divergent="RdBu_r",
    default_negative="Blues_r",
    default_positive="Reds",
    palette_type="light",
    legend_type="colorbar",
    edgecolor="white",
    palette_center=None,
    binning="uniform",
    alpha=1.0,
    linewidth=1,
    zorder=1,
    cbar_args={},
    **kwargs
):

    geodf = geodf[pd.notnull(geodf[column])].copy()
    min_value, max_value = geodf[column].min(), geodf[column].max()

    if binning in ("fisher_jenks", "quantiles"):
        if binning == "fisher_jenks":
            binning_method = FisherJenks(geodf[column], k=k)
        else:
            binning_method = Quantiles(geodf[column], k=k)
        bins = np.insert(binning_method.bins, 0, geodf[column].min())
        geodf = geodf.assign(__bin__=binning_method.yb)
    elif binning == "uniform":
        bins = np.linspace(
            min_value, max_value + (max_value - min_value) * 0.001, num=k + 1
        )
        geodf = geodf.assign(
            __bin__=lambda x: pd.cut(
                x[column], bins=bins, include_lowest=True, labels=False
            ).astype(np.int)
        )
    else:
        raise ValueError(
            "only fisher_jenks, quantiles and uniform binning are supported"
        )

    cmap_name = None
    norm = None
    midpoint = 0.0
    using_divergent = False
    built_palette = None

    if palette_center is not None:
        midpoint = palette_center

    if palette is None:
        if min_value < 0 and max_value > 0:
            # divergent
            cmap_name = default_divergent
            norm = MidpointNormalize(vmin=min_value, vmax=max_value, midpoint=midpoint)
            using_divergent = True
        elif min_value < 0:
            # sequential
            cmap_name = default_negative
        else:
            # sequential
            cmap_name = default_positive
    else:
        if not isinstance(palette, colors.Colormap):
            if colors.is_color_like(palette):
                if palette_type == "light":
                    built_palette = sns.light_palette(palette, n_colors=k)
                else:
                    built_palette = sns.dark_palette(palette, n_colors=k)
            else:
                built_palette = sns.color_palette(palette, n_colors=k)

    if norm is None:
        if palette_center is None:
            # norm = colors.Normalize(vmin=min_value, vmax=max_value)
            norm = colors.BoundaryNorm(bins, k)
        else:
            norm = MidpointNormalize(vmin=min_value, vmax=max_value, midpoint=midpoint)
            using_divergent = True

    if built_palette is None:
        if not using_divergent:
            built_palette = sns.color_palette(cmap_name, n_colors=k)
        else:
            middle_idx = np.where((bins[:-1] * bins[1:]) < 0)[0][0]
            left = middle_idx
            right = k - middle_idx - 1
            if left == right:
                built_palette = sns.color_palette(cmap_name, n_colors=k)
            else:
                delta = np.abs(left - right)
                expanded_k = k + delta
                start_idx = max(middle_idx - left, right - middle_idx)
                built_palette = sns.color_palette(cmap_name, n_colors=expanded_k)[
                    start_idx : start_idx + k
                ]

    # if legend_type == "hist":
    #     color_legend(
    #         cbar_ax,
    #         built_palette,
    #         bins,
    #         sizes=np.histogram(geodf[column], bins=bins)[0],
    #         orientation=cbar_orientation,
    #         remove_axes=False,
    #     )
    # elif legend_type == "colorbar":
    #     color_legend(
    #         cbar_ax,
    #         built_palette,
    #         bins,
    #         orientation=cbar_orientation,
    #         remove_axes=False,
    #     )

    for idx, group in geodf.groupby("__bin__"):
        group.plot(
            ax=ax,
            facecolor=built_palette[idx],
            linewidth=linewidth,
            edgecolor=edgecolor,
            alpha=alpha,
            zorder=zorder,
        )

    cbar_ax = add_ranged_color_legend(ax, bins, built_palette, **cbar_args)

    return ax, cbar_ax
