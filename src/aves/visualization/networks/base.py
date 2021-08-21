import graph_tool
import graph_tool.draw
import graph_tool.inference
import graph_tool.topology

from aves.models.network import Network

from .edges import *
from .fdeb import FDB
from .heb import HierarchicalEdgeBundling
from .nodes import *


class NodeLink(object):
    def __init__(self, network: Network):
        self.network = network
        self.bundle_model = None
        self.edge_strategy: EdgeStrategy = None
        self.edge_strategy_args: dict = None

        self.node_strategy: NodeStrategy = None
        self.node_strategy_args: dict = None

        self.labels = None

    def layout_nodes(self, *args, **kwargs):
        self.network.layout_nodes(*args, **kwargs)
        self._maybe_update_strategies()

    def _maybe_update_strategies(self, nodes=True, edges=True):
        if edges and self.edge_strategy is not None:
            self.set_edge_drawing(
                method=self.edge_strategy.name(), **self.edge_strategy_args
            )

        if nodes and self.node_strategy is not None:
            self.set_node_drawing(
                method=self.node_strategy.name(), **self.node_strategy_args
            )

    def plot_edges(self, ax, *args, **kwargs):
        if self.edge_strategy is None:
            self.set_edge_drawing()

        self.edge_strategy.plot(ax, *args, **kwargs)

    def plot_nodes(self, ax, *args, **kwargs):
        if self.node_strategy is None:
            self.set_node_drawing()

        self.node_strategy.plot(ax, *args, **kwargs)

    def plot(self, ax, *args, **kwargs):
        nodes = kwargs.get("nodes", {})
        edges = kwargs.get("edges", {})
        zorder = kwargs.get("zorder", None)
        self.plot_edges(ax, zorder=zorder, **edges)
        self.plot_nodes(ax, zorder=zorder, **nodes)

    def bundle_edges(self, method: str, *args, **kwargs):
        if method == "force-directed":
            self.bundle_model = FDB(self.network, *args, **kwargs)
        elif method == "hierarchical":
            self.bundle_model = HierarchicalEdgeBundling(self.network, *args, **kwargs)
        else:
            raise ValueError(f"method {str} not supported")

        self._maybe_update_strategies()

        return self.bundle_model

    def set_edge_drawing(self, method="plain", **kwargs):
        if method == "weighted":
            self.edge_strategy = WeightedEdges(
                self.network.edge_data, kwargs.get("k", 5)
            )
        elif method == "origin-destination":
            if not self.network.is_directed():
                raise ValueError("method only works with directed graphs")
            self.edge_strategy = ODGradient(
                self.network.edge_data, kwargs.get("n_points", 30)
            )
        elif method == "community-gradient":
            if type(self.bundle_model) != HierarchicalEdgeBundling:
                raise ValueError(f"{method} only works with HierarchicalEdgeBundling")
            level = kwargs.get("level", 1)
            communities = self.bundle_model.get_node_memberships(level)
            self.edge_strategy = CommunityGradient(
                self.network.edge_data, node_communities=communities
            )
        elif method == "plain":
            self.edge_strategy = PlainEdges(self.network.edge_data, **kwargs)
        else:
            raise ValueError(f"{method} is not supported")

        self.edge_strategy_args = dict(**kwargs)

    def set_node_drawing(self, method="plain", **kwargs):
        if method == "plain":
            self.node_strategy = PlainNodes(self.network, **kwargs)
        elif method == "labeled":
            if self.labels is None:
                raise ValueError("labeled strategy requires set up labels")
            self.node_strategy = LabeledNodes(self.network, self.labels, **kwargs)
        else:
            raise ValueError(f"{method} is not supported")

        self.node_strategy_args = dict(**kwargs)

    def set_node_labels(self, func=None):
        graph = self.network.graph()
        self.labels = graph.new_vertex_property("string")

        for idx in graph.vertices():
            if func is not None:
                self.labels[idx] = func(self.network, idx)
            else:
                self.labels[idx] = f"{self.network.id_to_label[int(idx)]}"

        self._maybe_update_strategies(edges=False)
