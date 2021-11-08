import contextily as cx
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colorbar
from seaborn.axisgrid import FacetGrid, Grid

from aves.visualization.colors import colormap_from_palette
from aves.visualization.maps.utils import geographical_scale, north_arrow


class GeoFacetGrid(FacetGrid):
    def __init__(self, geodataframe: gpd.GeoDataFrame, *args, **kwargs):
        geocontext = kwargs.pop("context", None)

        if geocontext is None:
            geocontext = geodataframe

        self.geocontext = geocontext

        self.bounds = geocontext.total_bounds
        self.aspect = (self.bounds[2] - self.bounds[0]) / (
            self.bounds[3] - self.bounds[1]
        )

        kwargs["aspect"] = self.aspect
        kwargs["xlim"] = (self.bounds[0], self.bounds[2])
        kwargs["ylim"] = (self.bounds[1], self.bounds[3])

        super().__init__(geodataframe, *args, **kwargs)

        for ax in self.axes.flatten():
            if kwargs.get("remove_axes", True):
                ax.set_axis_off()
            if kwargs.get("equal_aspect", True):
                ax.set_aspect("equal")

        self.zorder = 0

    def add_layer(self, func_or_data, *args, **kwargs):
        if isinstance(func_or_data, gpd.GeoDataFrame):
            # a direct geography
            for ax in self.axes.flatten():
                func_or_data.plot(*args, ax=ax, zorder=self.zorder, **kwargs)
        else:
            plot = lambda *a, **kw: func_or_data(
                plt.gca(), kw.pop("data"), *a, zorder=self.zorder, **kw
            )
            self.map_dataframe(plot, *args, **kwargs)

        self.zorder += 1

    def add_basemap(
        self, file_path, interpolation="hanning", reset_extent=False, **kwargs
    ):
        for ax in self.axes.flatten():
            cx.add_basemap(
                ax,
                crs=self.geocontext.crs.to_string(),
                source=file_path,
                interpolation=interpolation,
                zorder=self.zorder,
                reset_extent=reset_extent,
                **kwargs
            )

            if not reset_extent:
                ax.set_xlim((self.bounds[0], self.bounds[2]))
                ax.set_ylim((self.bounds[1], self.bounds[3]))

        self.zorder += 1

    def add_map_elements(
        self,
        all_axes=False,
        scale=True,
        arrow=True,
        scale_args={},
        arrow_args={},
    ):
        for ax in self.axes.flatten():
            if arrow:
                north_arrow(ax, **arrow_args)

            if scale:
                geographical_scale(ax, **scale_args)
            if not all_axes:
                break

    def add_global_colorbar(self, palette, k, title=None, title_args={}, **kwargs):
        orientation = kwargs.get("orientation", "horizontal")

        if orientation == "horizontal":
            cax = self.fig.add_axes([0.25, -0.012, 0.5, 0.01])
        elif orientation == "vertical":
            cax = self.fig.add_axes([1.01, 0.25, 0.01, 0.5])
        else:
            raise ValueError("unsupported colorbar orientation")

        if title:
            cax.set_title(title, **title_args)

        cax.set_axis_off()
        cb = colorbar.ColorbarBase(
            cax, cmap=colormap_from_palette(palette, n_colors=k), **kwargs
        )

        return cax, cb

    def set_title(self, title, **kwargs):
        self.fig.suptitle(title, **kwargs)


class GeoAttributeGrid(Grid):
    def __init__(
        self,
        geodataframe: gpd.GeoDataFrame,
        *,
        context: gpd.GeoDataFrame = None,
        vars=None,
        height=2.5,
        layout_pad=0.5,
        col_wrap=4,
        despine=True,
        remove_axes=True,
        set_limits=True,
        equal_aspect=True
    ):

        super().__init__()

        if vars is not None:
            vars = list(vars)
        else:
            vars = list(geodataframe.drop("geometry", axis=1).columns)

        if not vars:
            raise ValueError("No variables found for grid.")

        self.vars = vars

        if context is None:
            geocontext = geodataframe

        self.geocontext = geocontext

        self.bounds = geocontext.total_bounds
        self.aspect = (self.bounds[2] - self.bounds[0]) / (
            self.bounds[3] - self.bounds[1]
        )

        n_variables = len(vars)

        n_columns = min(col_wrap, len(vars))
        n_rows = n_variables // n_columns
        if n_rows * n_columns < n_variables:
            n_rows += 1

        with mpl.rc_context({"figure.autolayout": False}):
            fig, axes = plt.subplots(
                n_rows,
                n_columns,
                figsize=(n_columns * height * self.aspect, n_rows * height),
                sharex=True,
                sharey=True,
                squeeze=False,
            )

        flattened = axes.flatten()

        if set_limits:
            for ax in flattened:
                ax.set_xlim((self.bounds[0], self.bounds[2]))
                ax.set_ylim((self.bounds[1], self.bounds[3]))

        if remove_axes:
            for ax in flattened:
                ax.set_axis_off()
        else:
            # deactivate only unneeded axes
            for i in range(n_variables, len(axes)):
                flattened[i].set_axis_off()

        if equal_aspect:
            for ax in flattened:
                ax.set_aspect("equal")

        self._figure = fig
        self.axes = axes
        self.data = geodataframe

        # Label the axes
        self._add_axis_labels()

        self._legend_data = {}

        # Make the plot look nice
        self._tight_layout_rect = [0.01, 0.01, 0.99, 0.99]
        self._tight_layout_pad = layout_pad
        self._despine = despine
        if despine:
            sns.despine(fig=fig)
        self.tight_layout(pad=layout_pad)

    def _add_axis_labels(self):
        for ax, label in zip(self.axes.flatten(), self.vars):
            ax.set_title(label)

    def add_layer(self, func_or_data, *args, **kwargs):
        if isinstance(func_or_data, gpd.GeoDataFrame):
            # a direct geography
            for ax, col in zip(self.axes.flatten(), self.vars):
                func_or_data.plot(*args, ax=ax, **kwargs)
        else:
            for ax, col in zip(self.axes.flatten(), self.vars):
                func_or_data(ax, self.data, col, *args, **kwargs)


def figure_from_geodataframe(
    geodf,
    height=5,
    bbox=None,
    remove_axes=True,
    set_limits=True,
    basemap=None,
    basemap_interpolation="hanning",
):
    if bbox is None:
        bbox = geodf.total_bounds

    aspect = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
    fig, ax = plt.subplots(figsize=(height * aspect, height))

    if set_limits:
        ax.set_xlim([bbox[0], bbox[2]])
        ax.set_ylim([bbox[1], bbox[3]])

    if remove_axes:
        ax.set_axis_off()

    if basemap is not None:
        cx.add_basemap(
            ax,
            crs=geodf.crs.to_string(),
            source=basemap,
            interpolation=basemap_interpolation,
            zorder=0,
        )

    return fig, ax


def small_multiples_from_geodataframe(
    geodf,
    n_variables,
    height=5,
    col_wrap=5,
    bbox=None,
    sharex=True,
    sharey=True,
    remove_axes=True,
    set_limits=True,
    flatten_axes=True,
    equal_aspect=True,
    basemap=None,
    basemap_interpolation="hanning",
):
    if n_variables <= 1:
        return figure_from_geodataframe(
            geodf,
            height=height,
            bbox=bbox,
            remove_axes=remove_axes,
            set_limits=set_limits,
            basemap=basemap,
            basemap_interpolation=basemap_interpolation,
        )

    if bbox is None:
        bbox = geodf.total_bounds

    aspect = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])

    n_columns = min(col_wrap, n_variables)
    n_rows = n_variables // n_columns
    if n_rows * n_columns < n_variables:
        n_rows += 1

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(n_columns * height * aspect, n_rows * height),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )
    flattened = axes.flatten()

    if set_limits:
        for ax in flattened:
            ax.set_xlim([bbox[0], bbox[2]])
            ax.set_ylim([bbox[1], bbox[3]])

    if remove_axes:
        for ax in flattened:
            ax.set_axis_off()
    else:
        # deactivate only unneeded axes
        for i in range(n_variables, len(axes)):
            flattened[i].set_axis_off()

    if equal_aspect:
        for ax in flattened:
            ax.set_aspect("equal")

    if basemap is not None:
        for ax in flattened:
            cx.add_basemap(
                ax,
                crs=geodf.crs.to_string(),
                source=basemap,
                interpolation=basemap_interpolation,
                zorder=0,
            )

    if flatten_axes:
        return fig, flattened

    return fig, axes
