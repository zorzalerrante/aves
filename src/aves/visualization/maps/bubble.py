import geopandas as gpd


def bubble_map(
    ax,
    geodf: gpd.GeoDataFrame,
    size,
    scale=1,
    palette=None,
    color=None,
    add_legend=True,
    edgecolor="white",
    alpha=1.0,
    label=None,
    **kwargs
):
    marker = "o"

    if size is not None:
        if type(size) == str:
            marker_size = geodf[size]
        else:
            marker_size = float(size)
    else:
        marker_size = 1

    return geodf.plot(
        ax=ax,
        marker=marker,
        markersize=marker_size * scale,
        edgecolor=edgecolor,
        alpha=alpha,
        facecolor=color,
        legend=add_legend,
        label=label,
        **kwargs
    )


def dot_map(
    ax,
    geodf: gpd.GeoDataFrame,
    size=10,
    palette=None,
    add_legend=True,
    label=None,
    **kwargs
):
    return bubble_map(
        ax,
        geodf,
        size=float(size),
        palette=palette,
        add_legend=add_legend,
        edgecolor="none",
        label=label,
        **kwargs
    )
