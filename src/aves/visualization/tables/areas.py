import numpy as np
from aves.visualization.collections import LabelCollection
from aves.features.utils import tfidf, normalize_rows, normalize_columns


def stacked_areas(ax, df, baseline="zero", color_dict=None, **kwargs):
    stack = np.cumsum(df.T.values, axis=0)
    x = df.index.values
    y = df.T.values
    m = y.shape[0]

    if baseline == "zero":
        first_line = np.zeros(x.shape)
    else:
        first_line = (y * (m - 0.5 - np.arange(m)[:, None])).sum(0)
        first_line /= -m
        first_line -= np.min(first_line)
        stack += first_line

    color = color_dict[df.columns[0]] if color_dict is not None else None
    ax.fill_between(x, first_line, stack[0, :], facecolor=color, **kwargs)

    for i in range(len(y) - 1):

        color = color_dict[df.columns[i + 1]] if color_dict is not None else None
        ax.fill_between(x, stack[i, :], stack[i + 1, :], facecolor=color, **kwargs)

    return x, first_line, stack


def streamgraph(
    ax,
    df,
    baseline="wiggle",
    labels=True,
    label_threshold=0,
    label_args=None,
    fig=None,
    area_colors=None,
    area_args=None,
    avoid_label_collisions=False,
    outline_labels=True,
    label_collision_args=None,
):
    if label_args is None:
        label_args = {}

    if area_args is None:
        area_args = {}

    if label_collision_args is None:
        label_collision_args = dict(
            lim=25, arrowprops=dict(arrowstyle="-", color="k", lw=0.5)
        )

    df = df.fillna(0).astype(np.float)

    stream_x, stream_first_line, stream_stack = stacked_areas(
        ax, df, color_dict=area_colors, baseline=baseline, **area_args
    )

    if labels:
        x_to_idx = dict(zip(df.index, range(len(df))))
        max_x = df.idxmax().map(x_to_idx)

        label_collection = LabelCollection()

        max_idx = max_x.values[0]
        y_value = stream_stack[0, max_idx] - stream_first_line[max_idx]

        if y_value >= label_threshold:
            label_collection.add_text(
                df.columns[0],
                stream_x[max_idx],
                stream_first_line[max_idx] * 0.5 + stream_stack[0, max_idx] * 0.5,
            )

        for i in range(1, len(df.columns)):
            max_idx = max_x.values[i]
            y_value = stream_stack[i, max_idx] - stream_stack[i - 1, max_idx]

            if y_value < label_threshold:
                continue

            label_collection.add_text(
                df.columns[i],
                stream_x[max_idx],
                stream_stack[i, max_idx] * 0.5 + stream_stack[i - 1, max_idx] * 0.5,
            )

        label_collection.render(
            ax,
            fig=fig,
            color=label_args.get("color", "white"),
            fontweight=label_args.get("fontweight", "bold"),
            fontsize=label_args.get("fontsize", "xx-large"),
            outline=outline_labels,
            avoid_collisions=avoid_label_collisions,
            adjustment_args=label_collision_args,
        )
