from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cytoolz import valmap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from .base import DataFusionModel


def fusion_diagram(
    model: DataFusionModel,
    margin=2,
    height=14,
    facecolor="#FFB7C5",
    transform=None,
    sort_metric="correlation",
):
    table = pd.DataFrame.from_records(
        list(
            map(
                lambda l: list(chain(*l)),
                sorted(
                    valmap(lambda x: x[0].shape, model.relation_definitions).items()
                ),
            )
        )
    )
    table.columns = ["src", "dst", "rows", "columns"]

    if transform is not None:
        table["rows"] = np.ceil(transform(table["rows"]))
        table["columns"] = np.ceil(transform(table["columns"]))

    row_sizes = table.pivot(index="src", columns="dst", values="rows") + margin

    if sort_metric == "euclidean":
        method = "ward"
    else:
        method = "average"

    np.random.seed(98)
    g = sns.clustermap(
        pd.notnull(row_sizes).astype(int),
        metric=sort_metric,
        method=method,
        figsize=(1, 1),
    )
    plt.clf()

    row_sizes = row_sizes.loc[g.data2d.index][g.data2d.columns]
    column_sizes = table.pivot(index="src", columns="dst", values="columns") + margin
    column_sizes = column_sizes.loc[g.data2d.index][g.data2d.columns]

    total_rows = row_sizes.max(axis=1).sum()
    total_columns = column_sizes.max(axis=0).sum()

    start_columns = (
        (column_sizes.max(axis=0).cumsum() - column_sizes.max(axis=0))
        .astype(int)
        .to_dict()
    )
    start_rows = (
        (row_sizes.max(axis=1).cumsum() - row_sizes.max(axis=1)).astype(int).to_dict()
    )

    fig, ax = plt.subplots(figsize=(height * total_columns / total_rows, height))
    ax.set_xlim([-margin, total_columns])
    ax.set_ylim([total_rows, -margin])

    boxes = []

    ax.axhline(-margin / 2, color="#abacab", linestyle="dotted", linewidth=1)
    ax.axvline(-margin / 2, color="#abacab", linestyle="dotted", linewidth=1)

    annotated_x = set()
    annotated_y = set()

    text_margin = margin * 0.05

    for idx, row in table.iterrows():
        # print(row)
        box = Rectangle(
            (start_columns[row["dst"]], start_rows[row["src"]]),
            row["columns"],
            row["rows"],
        )
        boxes.append(box)

        ax.axhline(
            start_rows[row["src"]] + row["rows"] + margin / 2,
            color="#abacab",
            linestyle="dotted",
            linewidth=1,
        )
        ax.axvline(
            start_columns[row["dst"]] + row["columns"] + margin / 2,
            color="#abacab",
            linestyle="dotted",
            linewidth=1,
        )

        mid_y = start_rows[row["src"]] + 0.5 * row["rows"]
        mid_x = start_columns[row["dst"]] + 0.5 * row["columns"]

        if not row["dst"] in annotated_x:
            ax.text(
                mid_x, -margin / 2 - text_margin, row["dst"], ha="center", va="bottom"
            )
            annotated_x.add(row["dst"])

        if not row["src"] in annotated_y:
            ax.text(
                -margin / 2 - text_margin, mid_y, row["src"], ha="right", va="center"
            )
            annotated_y.add(row["src"])

    collection = PatchCollection(boxes, facecolor=facecolor, edgecolor="black")
    ax.add_collection(collection)
    ax.set_axis_off()

    return fig, ax
