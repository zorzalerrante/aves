import matplotlib.colors as colors
import numpy as np
import seaborn as sns
import spectra
from matplotlib import colorbar
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# from http://chris35wills.github.io/matplotlib_diverging_colorbar/


# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def colormap_from_palette(palette_name, n_colors=10):
    return colors.ListedColormap(sns.color_palette(palette_name, n_colors=n_colors))


def color_legend(
    ax,
    color_list,
    bins=None,
    sizes=None,
    orientation="horizontal",
    remove_axes=False,
    bin_spacing="proportional",
    tick_labels=None,
):
    if bins is None:
        if type(color_list) == colors.ListedColormap:
            N = color_list.N
        else:
            N = len(color_list)
        bins = np.array(range(N))

    if sizes is not None:
        bar_width = bins[1:] - bins[0:-1]
        if orientation == "horizontal":
            ax.bar(
                bins[:-1],
                sizes,
                width=bar_width,
                align="edge",
                color=color_list,
                edgecolor=color_list,
            )
            ax.set_xticks(bins, labels=tick_labels)
        else:
            ax.barh(
                bins[:-1],
                sizes,
                height=bar_width,
                align="edge",
                color=color_list,
                edgecolor=color_list,
            )
            ax.set_yticks(bins, labels=tick_labels)
    else:
        cbar_norm = colors.BoundaryNorm(bins, len(bins) - 1)
        if type(color_list) == colors.ListedColormap:
            cmap = color_list
        else:
            cmap = colors.ListedColormap(color_list)
        cb = colorbar.ColorbarBase(
            ax,
            cmap=cmap,
            norm=cbar_norm,
            ticks=bins,
            spacing=bin_spacing,
            orientation=orientation,
        )
        if tick_labels:
            cb.set_ticklabels(tick_labels)

    sns.despine(ax=ax, top=True, bottom=True, left=True, right=True)

    if remove_axes:
        ax.set_axis_off()

    return ax


def bivariate_matrix_from_palette(
    palette_name="PiYG", n_colors=3, darken_initial_value=-15
):
    full_palette = sns.color_palette(palette_name, n_colors=(n_colors - 1) * 2 + 1)

    cmap_x = full_palette[n_colors - 1 :]
    cmap_y = list(reversed(full_palette))[n_colors - 1 :]

    cmap_xy = []

    for j in range(n_colors):
        for i in range(n_colors):
            x = spectra.rgb(*cmap_x[i][0:3])
            y = spectra.rgb(*cmap_y[j][0:3])

            if i == j and i == 0:
                cmap_xy.append(x.darken(darken_initial_value).rgb)

            elif i == 0:
                cmap_xy.append(y.rgb)
            elif j == 0:
                cmap_xy.append(x.rgb)
            else:
                blended = x.blend(y, ratio=0.5).brighten(1.5)

                if i == j:
                    blended = blended
                else:
                    blended = blended.saturate(3)

                cmap_xy.append(blended.rgb)

    cmap_xy = np.array(cmap_xy).reshape(n_colors, n_colors, 3)
    return cmap_xy


def categorical_color_legend(
    ax, color_list, labels, loc="best", n_columns=None, **kwargs
):
    legend_elements = []

    for label, color in zip(labels, color_list):
        legend_elements.append(Patch(facecolor=color, edgecolor="none", label=label))

    if n_columns is not None:
        if type(n_columns) != int:
            n_columns = len(color_list)
    else:
        n_columns = 1

    artist = ax.legend(
        handles=legend_elements, loc=loc, frameon=False, ncol=n_columns, **kwargs
    )
    ax.add_artist(artist)
    return artist


def add_ranged_color_legend(
    ax,
    bins,
    built_palette,
    location="lower center",
    orientation="horizontal",
    label=None,
    label_size="x-small",
    title_size="small",
    title_align="left",
    width="50%",
    height="5%",
    bbox_to_anchor=(0.0, 0.0, 0.95, 0.95),
    bbox_transform=None,
    tick_labels=None,
    **kwargs,
):
    if bbox_transform is None:
        bbox_transform = ax.transAxes

    cbar_ax = kwargs.pop("cbar_ax", None)

    if cbar_ax is None:
        if location != "out":
            cbar_ax = inset_axes(
                ax,
                width=width,
                height=height,
                loc=location,
                bbox_to_anchor=bbox_to_anchor,
                bbox_transform=bbox_transform,
            )
        else:
            divider = make_axes_locatable(ax)
            cbar_main = divider.append_axes(
                "bottom" if orientation == "horizontal" else "right",
                size=height if orientation == "horizontal" else width,
            )
            cbar_main.set_axis_off()
            cbar_ax = inset_axes(
                cbar_main,
                width=width,
                height=height,
                loc="center",
                bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
                bbox_transform=cbar_main.transAxes,
                borderpad=0,
            )

    color_legend(
        cbar_ax,
        built_palette,
        bins,
        orientation=orientation,
        remove_axes=False,
        tick_labels=tick_labels if tick_labels is not None else [],
        **kwargs,
    )

    sns.despine(ax=cbar_ax, left=True, top=True, bottom=True, right=True)
    if label:
        cbar_ax.set_title(label, loc="left", fontsize=title_size)

    cbar_ax.tick_params(labelsize=label_size)

    return cbar_ax
