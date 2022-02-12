from typing import Dict, Optional

import graph_tool
import graph_tool.centrality
import graph_tool.inference
import graph_tool.topology
import numpy as np
import pandas as pd
from cytoolz import itemmap, keyfilter, valfilter

from .edge import Edge


class Network(object):
    def __init__(self, graph: graph_tool.Graph, edge_weight=None):
        self.network: graph_tool.Graph = graph
        self.edge_weight = edge_weight
        self.node_weight = None
        self.edge_data = None

        self.node_map: dict = None
        self.node_layout: LayoutStrategy = None

        self.id_to_label: dict = None

    @classmethod
    def from_edgelist(
        cls,
        df: pd.DataFrame,
        source="source",
        target="target",
        directed=True,
        weight=None,
    ):

        source_attr = f"{source}__mapped__"
        target_attr = f"{target}__mapped__"

        node_values = set(df[source].unique())
        node_values = node_values | set(df[target].unique())
        node_map = dict(zip(sorted(node_values), range(len(node_values))))

        df_mapped = df.assign(
            **{
                source_attr: df[source].map(node_map),
                target_attr: df[target].map(node_map),
            }
        )

        if weight is not None:
            network, edge_weight = cls._parse_edgelist(
                df_mapped,
                source=source_attr,
                target=target_attr,
                weight=weight,
                directed=directed,
            )
        else:
            network = cls._parse_edgelist(
                df_mapped,
                source=source_attr,
                target=target_attr,
                weight=None,
                directed=directed,
            )
            edge_weight = None

        result = cls(network, edge_weight=edge_weight)
        result.node_map = node_map
        result.id_to_label = itemmap(reversed, node_map)
        return result

    @classmethod
    def _parse_edgelist(
        cls,
        df,
        source="source",
        target="target",
        weight="weight",
        directed=True,
        remove_empty=True,
    ):
        network = graph_tool.Graph(directed=directed)
        n_vertices = max(df[source].max(), df[target].max()) + 1
        vertex_list = network.add_vertex(n_vertices)

        if weight is not None and weight in df.columns:
            if remove_empty:
                df = df[df[weight] > 0]
            weight_prop = network.new_edge_property("double")
            network.add_edge_list(
                df.assign(**{weight: df[weight].astype(np.float64)})[
                    [source, target, weight]
                ].values,
                eprops=[weight_prop],
            )
            # network.shrink_to_fit()
            return network, weight_prop
        else:
            network.add_edge_list(df[[source, target]].values)
            # network.shrink_to_fit()
            return network

    def build_edge_data(self):
        if self.edge_data is None:
            # first time?
            self.edge_data = []

            for i, e in enumerate(self.network.edges()):
                src_idx = int(e.source())
                dst_idx = int(e.target())
                if src_idx == dst_idx:
                    # no support for self connections yet
                    continue

                src = self.node_layout.get_position(src_idx)
                dst = self.node_layout.get_position(dst_idx)
                weight = self.edge_weight[e] if self.edge_weight is not None else 1

                edge = Edge(src, dst, src_idx, dst_idx, weight=weight, index=i)
                self.edge_data.append(edge)
        else:
            # update positions only! the rest may have been changed manually.
            for data in self.edge_data:
                src_idx = data.index_pair[0]
                dst_idx = data.index_pair[1]
                src = self.node_layout.get_position(src_idx)
                dst = self.node_layout.get_position(dst_idx)
                data.source = src
                data.target = dst
                data.points = [src, dst]

    def layout_nodes(self, *args, **kwargs):
        from .layouts import (
            ForceDirectedLayout,
            GeographicalLayout,
            PrecomputedLayout,
            RadialLayout,
        )

        method = kwargs.pop("method", "force-directed")

        if method == "force-directed":
            self.node_layout = ForceDirectedLayout(self)
        elif method == "precomputed":
            self.node_layout = PrecomputedLayout(self)
        elif method == "geographical":
            geodf = kwargs.pop("geodataframe")
            node_column = kwargs.pop("node_column", None)
            self.node_layout = GeographicalLayout(self, geodf, node_column=node_column)
        elif method == "radial":
            self.node_layout = RadialLayout(self)
        else:
            raise NotImplementedError(f"unknown layout method {method}")

        self.node_layout.layout_nodes(*args, **kwargs)
        self.build_edge_data()

    def num_vertices(self):
        return self.network.num_vertices()

    def num_edges(self):
        return self.network.num_edges()

    def vertices(self):
        return self.network.vertices()

    def edges(self):
        return self.network.edges()

    def is_directed(self):
        return self.network.is_directed()

    def graph(self):
        return self.network

    def shortest_path(self, src, dst):
        paths = list(
            graph_tool.topology.all_shortest_paths(
                self.network, self.node_map[src], self.node_map[dst]
            )
        )

        return [[self.id_to_label[v] for v in path] for path in paths]

    def subgraph(
        self, nodes=None, vertex_filter=None, edge_filter=None, copy_positions=True
    ):
        if nodes is not None:
            view = graph_tool.GraphView(
                self.network, vfilt=lambda x: self.id_to_label[x] in nodes
            ).copy()
        elif vertex_filter is not None or edge_filter is not None:
            view = graph_tool.GraphView(
                self.network, vfilt=vertex_filter, efilt=edge_filter
            ).copy()
        else:
            raise ValueError("at least one filter must be specified")

        old_vertex_ids = set(map(int, view.vertices()))

        if copy_positions and self.node_layout is not None:
            vertex_positions = [
                self.node_layout.get_position(v_id) for v_id in view.vertices()
            ]
        else:
            vertex_positions = None

        node_map_keys = valfilter(lambda x: x in old_vertex_ids, self.node_map).keys()

        # TODO: vertex ids also change
        # TODO: vertex weights

        if self.edge_weight is not None:
            edge_weight = [self.edge_weight[e_id] for e_id in view.edges()]
        else:
            edge_weight = None

        view.purge_vertices()
        view.purge_edges()

        if edge_weight is not None:
            weight_prop = view.new_edge_property("double")
            weight_prop.a = edge_weight
        else:
            weight_prop = None

        result = Network(view, edge_weight=weight_prop)
        result.node_map = dict(zip(node_map_keys, map(int, view.vertices())))
        # print(result.node_map)
        result.id_to_label = itemmap(reversed, result.node_map)

        if vertex_positions:
            result.layout_nodes(method="precomputed", positions=vertex_positions)
        return result

    def node_degree(self, degree_type="in"):
        if not degree_type in ("in", "out", "total"):
            raise ValueError("Unsupported node degree")

        return getattr(self.network, f"get_{degree_type}_degrees")(
            list(self.network.vertices()), eweight=self.edge_weight
        )

    def get_betweenness(self, update_nodes=False, update_edges=False):
        node_centrality, edge_centrality = graph_tool.centrality.betweenness(
            self.network
        )

        if update_edges:
            for edge in self.edge_data:
                edge.weight = edge_centrality[self.network.edge(*edge.index_pair)]

        if update_nodes:
            self.node_weight = node_centrality

        return node_centrality, edge_centrality

    def connected_components(self, directed=True):
        return graph_tool.topology.label_components(self.network, directed=directed)

    def largest_connected_component(self, directed=True):
        components = self.connected_components(directed=directed)
        view = self.subgraph(
            vertex_filter=lambda x: components[0][x] == np.argmax(components[1])
        )
        return view
