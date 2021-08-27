import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.weightstats import DescrStatsW
from aves.visualization.colors import categorical_color_legend


def boxplot_stats(values: pd.Series, weights: pd.Series, label=None):
    w = DescrStatsW(values, weights)
    w_quants = w.quantile([0.25, 0.5, 0.75])
    w_iqr = w_quants[0.75] - w_quants[0.25]
    w_whisker_low = w_quants[0.25] - 1.5 * w_iqr
    w_whisker_high = w_quants[0.75] + 1.5 * w_iqr
    w_fliers = values[~values.between(w_whisker_low, w_whisker_high)]

    if values.min() > w_whisker_low:
        w_whisker_low = values.min()

    if values.max() < w_whisker_high:
        w_whisker_high = values.max()

    result = {
        "med": w_quants[0.5],
        "q1": w_quants[0.25],
        "q3": w_quants[0.75],
        "whislo": w_whisker_low,
        "whishi": w_whisker_high,
        "fliers": w_fliers,
    }

    if label is not None:
        result["label"] = label

    return pd.Series(result)


def boxplot(
    ax,
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    weight_column: str,
    hue_column=None,
    sort_by_value=False,
    sort_ascending=True,
    hue_order=None,
    vert=True,
    showfliers=False,
    palette="Set2",
    hue_legend=False,
    boxplot_kwargs={},
    legend_kwargs={},
):
    if not "boxprops" in boxplot_kwargs:
        boxplot_kwargs["boxprops"] = {}

    if not "medianprops" in boxplot_kwargs:
        boxplot_kwargs["medianprops"] = dict(color="black")

    if not "flierprops" in boxplot_kwargs:
        boxplot_kwargs["flierprops"] = dict(
            color="#abacab", marker=".", markersize="1", alpha=0.5
        )

    if hue_column is None:
        if not "facecolor" in boxplot_kwargs["boxprops"]:
            boxplot_kwargs["boxprops"]["facecolor"] = "#efefef"

        grouped = df.groupby(group_column).apply(
            lambda x: boxplot_stats(
                x[value_column], x[weight_column], x[group_column].values[0]
            )
        )

        if sort_by_value:
            grouped = grouped.sort_values("med", ascending=sort_ascending)

        grouped = grouped.to_dict(orient="records")

        ax.bxp(
            grouped,
            showfliers=showfliers,
            vert=vert,
            patch_artist=True,
            **boxplot_kwargs
        )
    else:
        if hue_order is None:
            hue_values = list(reversed(df[hue_column].unique()))
        else:
            hue_values = hue_order

        colors = sns.color_palette(palette, n_colors=len(hue_values))

        grouped = df.groupby([hue_column, group_column]).apply(
            lambda x: boxplot_stats(x[value_column], x[weight_column])
        )

        order = None
        categories = df[group_column].unique()
        width = 0.95 / len(hue_values)

        positions = (
            np.arange(len(categories)) + np.repeat(2 * width, len(categories)).cumsum()
        )

        offset = np.linspace(width * 0.25, 1 - width * 0.25, len(hue_values))
        offset -= offset.mean()
        # print(width, offset, positions)

        for i, hue in enumerate(hue_values):
            boxplot_kwargs["boxprops"]["facecolor"] = colors[i]

            group = grouped.loc[hue]

            if order is None:
                if sort_by_value:
                    group = group.sort_values("med", ascending=sort_ascending)

                order = list(group.index.values)
                if len(order) != len(categories):
                    order.extend([c for c in categories if not c in order])

            # enforces the order and inserts potentially missing rows
            group = pd.DataFrame(index=order).join(group, how="left").copy()

            for j, category in enumerate(order):
                if not category in group.index:
                    continue

                records = group.loc[category].to_dict()

                ax.bxp(
                    [records],
                    positions=[positions[j] - offset[i]],
                    showfliers=showfliers,
                    vert=vert,
                    patch_artist=True,
                    widths=[width],
                    **boxplot_kwargs
                )

        if vert:
            ax.set_xticks(positions)
            ax.set_xticklabels(order)
            ax.set_xlim([-offset[0], len(categories) + offset[-1]])
            ax.set_xlim(
                [
                    positions[0] + offset[0] - width * 1.5,
                    positions[-1] + offset[-1] + width * 1.5,
                ]
            )
        else:
            ax.set_yticks(positions)
            ax.set_yticklabels(order)
            ax.set_ylim(
                [
                    positions[0] + offset[0] - width * 1.5,
                    positions[-1] + offset[-1] + width * 1.5,
                ]
            )

        if hue_legend:
            categorical_color_legend(ax, colors, hue_values, **legend_kwargs)
