import matplotlib.patheffects as path_effects
from matplotlib_scalebar.scalebar import (
    ScaleBar,
)
import contextily as cx


def geographical_labels(
    ax,
    geodf,
    column=None,
    font_size=11,
    font_weight="bold",
    color="white",
    outline=True,
    outline_color="black",
    outline_width=2,
    format_func=None,
    bounds=None,
):
    labels = []
    for idx, row in geodf.iterrows():
        centroid = row.geometry.centroid

        if bounds is not None:
            if not (bounds[0] <= centroid.x <= bounds[2]) or not (
                bounds[1] <= centroid.y <= bounds[3]
            ):
                continue
        if column is None:
            label = idx
        else:
            label = row[column]

        if format_func is not None:
            label = format_func(label)

        t = ax.text(
            centroid.x,
            centroid.y,
            label,
            va="center",
            horizontalalignment="center",
            fontsize=font_size,
            fontweight=font_weight,
            color=color,
        )
        if outline:
            t.set_path_effects(
                [
                    path_effects.Stroke(
                        linewidth=outline_width, foreground=outline_color
                    ),
                    path_effects.Normal(),
                ]
            )
        labels.append(t)

    return labels


def north_arrow(
    ax,
    x=0.98,
    y=0.06,
    arrow_length=0.04,
    text="N",
    font_name=None,
    font_size=None,
    color="#000000",
    arrow_color="#000000",
    arrow_width=3,
    arrow_headwidth=7,
):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x, y - arrow_length),
        arrowprops=dict(
            facecolor=arrow_color, width=arrow_width, headwidth=arrow_headwidth
        ),
        ha="center",
        va="center",
        fontsize=font_size,
        fontname=font_name,
        color=color,
        xycoords=ax.transAxes,
    )


def geographical_scale(ax, location="lower left"):
    ax.add_artist(ScaleBar(1, location="lower left"))


def add_basemap(
    ax,
    file_path,
    geocontext,
    interpolation="hanning",
    zorder=0,
    reset_extent=False,
    **kwargs
):
    bounds = geocontext.total_bounds
    cx.add_basemap(
        ax,
        crs=geocontext.crs.to_string(),
        source=file_path,
        interpolation=interpolation,
        zorder=zorder,
        reset_extent=reset_extent,
        **kwargs
    )

    if not reset_extent:
        ax.set_xlim((bounds[0], bounds[2]))
        ax.set_ylim((bounds[1], bounds[3]))
