import matplotlib.pyplot as plt
import seaborn as sns

from aves.features.utils import normalize_rows


def barchart(
    ax,
    df,
    palette="plasma",
    stacked=False,
    normalize=False,
    sort_items=False,
    sort_categories=False,
    fill_na_value=None,
    bar_width=0.9,
    legend=True,
    legend_args=None,
    return_df=False,
    **kwargs
):
    sns.set_palette(palette, n_colors=len(df.columns))

    if fill_na_value is not None:
        df = df.fillna(fill_na_value)

    if normalize:
        df = df.pipe(normalize_rows)

    if sort_categories:
        sort_values = df.mean(axis=0).sort_values(ascending=False)
        df = df[sort_values.index].copy()

    if sort_items:
        df = df.sort_values(df.columns[0])

    df.plot.bar(
        ax=ax,
        stacked=stacked,
        width=bar_width,
        edgecolor="none",
        legend=legend,
        **kwargs
    )

    if legend:
        if legend_args is None:
            legend_args = dict(
                bbox_to_anchor=(1.0, 0.5), loc="center left", frameon=False
            )
        handles, labels = map(reversed, ax.get_legend_handles_labels())
        ax.legend(handles, labels, **legend_args)

    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    sns.despine(ax=ax, left=True)

    if normalize:
        ax.set_ylim([0, 1])

    if return_df:
        return df
