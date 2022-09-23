from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from cytoolz import unique
from matplotlib.collections import LineCollection

from aves.models.network import Network
from aves.visualization.collections import ColoredCurveCollection
from aves.visualization.primitives import RenderStrategy


class EdgeStrategy(RenderStrategy):
    """An interface to methods to render edges"""

    def __init__(self, network: Network, **kwargs):
        self.network = network
        super().__init__(self.network.edge_data)

    def name(self):
        return "strategy-name"


class PlainEdges(EdgeStrategy):
    """All edges are rendered with the same color"""

    def __init__(self, network, **kwargs):
        super().__init__(network)
        self.lines = None

    def prepare_data(self):
        self.lines = []

        for edge in self.data:
            self.lines.append(edge.points)

        self.lines = np.array(self.lines)

    def render(
        self,
        ax,
        color="#abacab",
        linewidth=1.0,
        linestyle="solid",
        alpha=0.75,
        **kwargs,
    ):
        collection = LineCollection(
            self.lines,
            color=color,
            linewidths=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            **kwargs,
        )
        return ax.add_collection(collection)

    def name(self):
        return "plain"


class WeightedEdges(EdgeStrategy):
    """Colors encode edge weight"""

    def __init__(self, network, weights, k, **kwargs):
        super().__init__(network)
        # self.edge_data_per_group = {i: [] for i in range(k)}
        # self.strategy_per_group = {
        #    i: PlainEdges(self.edge_data_per_group[i]) for i in range(k)
        # }
        self.k = k
        self.bins = None
        self.weights = weights
        self.lines = None

    def prepare_data(self):
        self.lines = []

        for edge in self.data:
            self.lines.append(edge.points)

        self.lines = np.array(self.lines)

        weights = self.weights

        if type(weights) == str:
            if not weights in self.network.network.edge_properties:
                if weights == "betweenness":
                    self.network.estimate_betweenness()
                else:
                    raise Exception("weights must be a valid edge property if str")

            weights = np.array(self.network.network.edge_properties[weights].a)

        if weights is not None and not type(weights) in (np.array, np.ndarray):
            raise ValueError(f"weights must be np.array instead of {type(weights)}.")

        weights: np.array = weights

        groups, bins = pd.cut(weights, self.k, labels=False, retbins=True)
        self.bins = bins
        self.line_groups = groups

    def render(self, ax, *args, **kwargs):
        palette = kwargs.pop("palette", None)

        if palette is None:
            edge_colors = list(
                reversed(
                    sns.dark_palette(kwargs.pop("color", "#a7a7a7"), n_colors=self.k)
                )
            )
        else:
            edge_colors = sns.color_palette(palette, n_colors=self.k)

        results = []
        for i in range(self.k):
            coll_lines = self.lines[self.line_groups == i]

            coll = LineCollection(
                coll_lines,
                color=edge_colors[i],
                **kwargs,
            )
            results.append(ax.add_collection(coll))

        return results

    def name(self):
        return "weighted"


class CommunityGradient(EdgeStrategy):
    """Colors encode communities of each node at the edges"""

    def __init__(self, network, node_communities, **kwargs):
        super().__init__(network)
        self.node_communities = node_communities
        self.community_ids = sorted(unique(node_communities))
        self.community_links = defaultdict(ColoredCurveCollection)

    def prepare_data(self):
        for edge_data in self.data:
            pair = (
                self.node_communities[int(edge_data.index_pair[0])],
                self.node_communities[int(edge_data.index_pair[1])],
            )

            # TODO: add weight
            self.community_links[pair].add_curve(edge_data.points, 1)

    def render(self, ax, *args, **kwargs):
        community_colors = dict(
            zip(
                self.community_ids,
                sns.color_palette(
                    kwargs.pop("palette", "plasma"), n_colors=len(self.community_ids)
                ),
            )
        )

        for pair, colored_lines in self.community_links.items():
            colored_lines.set_colors(
                source=community_colors[pair[0]], target=community_colors[pair[1]]
            )
            colored_lines.render(ax, *args, **kwargs)

    def name(self):
        return "community-gradient"


class ODGradient(EdgeStrategy):
    """Colors encode direction of edges"""

    def __init__(self, network, n_points, **kwargs):
        super().__init__(network)
        self.n_points = n_points
        self.colored_curves = ColoredCurveCollection()

    def prepare_data(self):
        interp = np.linspace(0, 1, num=self.n_points, endpoint=True)

        for edge_data in self.data:
            if type(edge_data.points) == list and len(edge_data.points) == 2:
                points = np.array(
                    [
                        (edge_data.source * (1 - t) + edge_data.target * t)
                        for t in interp
                    ]
                )
            elif type(edge_data.points) == np.array and edge_data.points.shape[0] == 2:
                points = np.array(
                    [
                        (edge_data.source * (1 - t) + edge_data.target * t)
                        for t in interp
                    ]
                )
            else:
                points = edge_data.points

            # TODO: add weight
            self.colored_curves.add_curve(points, 1)

    def render(self, ax, *args, **kwargs):
        self.colored_curves.set_colors(
            source=kwargs.pop("source_color", "blue"),
            target=kwargs.pop("target_color", "red"),
        )
        self.colored_curves.render(ax, *args, **kwargs)

    def name(self):
        return "origin-destination"
