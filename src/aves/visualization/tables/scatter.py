import matplotlib.pyplot as plt
import seaborn as sns

from aves.visualization.collections.labels import LabelCollection


def scatterplot(
    ax,
    df,
    x,
    y,
    hue=None,
    annotate=False,
    avoid_collisions=False,
    scatter_args={},
    text_args={},
    na_value=0,
    drop_na=False,
    adjustment_args={"lim": 5},
    label_filter_func=None,
):
    if not drop_na:
        df = df.fillna(na_value)
    else:
        df = df.dropna()

    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, **scatter_args)
    ax.ticklabel_format(useOffset=False, style="plain")
    sns.despine(ax=ax)

    if annotate:
        collection = LabelCollection()

        if label_filter_func is not None:
            label_df = df.pipe(label_filter_func)
        else:
            label_df = df

        for index, row in label_df.iterrows():
            collection.add_text(index, row[x], row[y])

        collection.render(
            ax,
            avoid_collisions=avoid_collisions,
            adjustment_args=adjustment_args,
            **text_args
        )
