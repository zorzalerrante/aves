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

        for index, row in df.iterrows():
            collection.add_text(index.title(), row[x], row[y])

        collection.render(
            ax,
            avoid_collisions=avoid_collisions,
            adjustment_args=adjustment_args,
            **text_args
        )
