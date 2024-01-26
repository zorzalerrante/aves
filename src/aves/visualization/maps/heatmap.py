import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from aves.features.geo import kde_from_points
from aves.visualization.colors import color_legend, colormap_from_palette


def heat_map(
    ax,
    geodf,
    weight=None,
    low_threshold=0,
    max_threshold=1.0,
    n_levels=5,
    alpha=1.0,
    palette="magma",
    kernel="cosine",
    norm=2,
    bandwidth=1e-2,
    grid_points=2**6,
    return_heat=False,
    cbar_label=None,
    cbar_width=2.4,
    cbar_height=0.15,
    cbar_location="upper left",
    cbar_orientation="horizontal",
    cbar_pad=0.05,
    cbar_bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
    cbar_bbox_transform=None,
    legend_type="none",
    **kwargs
):
    heat = kde_from_points(
        geodf,
        kernel=kernel,
        norm=norm,
        bandwidth=bandwidth,
        grid_points=grid_points,
        weight_column=weight,
    )

    norm_heat = heat[2] / heat[2].max()

    if type(palette) == str:
        cmap = colormap_from_palette(palette, n_colors=n_levels)
    else:
        cmap = palette

    levels = np.linspace(low_threshold, max_threshold, n_levels)

    # TODO: this should be factorized into an utility function
    if legend_type == "colorbar":
        # add_ranged_color_legend(ax)
        if cbar_bbox_transform is None:
            cbar_bbox_transform = ax.transAxes

        if cbar_location != "out":
            cbar_ax = inset_axes(
                ax,
                width=cbar_width,
                height=cbar_height,
                loc=cbar_location,
                bbox_to_anchor=cbar_bbox_to_anchor,
                bbox_transform=cbar_bbox_transform,
                borderpad=0,
            )
        else:
            divider = make_axes_locatable(ax)
            cbar_main = divider.append_axes(
                "bottom" if cbar_orientation == "horizontal" else "right",
                size=cbar_height if cbar_orientation == "horizontal" else cbar_width,
                pad=cbar_pad,
            )
            cbar_main.set_axis_off()
            cbar_ax = inset_axes(
                cbar_main,
                width=cbar_width,
                height=cbar_height,
                loc="center",
                bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
                bbox_transform=cbar_main.transAxes,
                borderpad=0,
            )

        color_legend(cbar_ax, cmap, levels, orientation=cbar_orientation)
    else:
        cbar_ax = None

    if not return_heat:
        return (
            ax.contourf(heat[0], heat[1], norm_heat, levels, alpha=alpha, cmap=cmap),
            cbar_ax,
        )
    else:
        return (
            ax.contourf(heat[0], heat[1], norm_heat, levels, alpha=alpha, cmap=cmap),
            cbar_ax,
            heat,
        )
