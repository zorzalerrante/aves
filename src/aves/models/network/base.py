from typing import Dict, Optional

import graph_tool
import graph_tool.centrality
import graph_tool.inference
import graph_tool.topology
import graph_tool.search
import numpy as np
import pandas as pd
from cytoolz import itemmap, valfilter, valmap
import joblib
from collections import defaultdict
import os.path
from pathlib import Path

from .edge import Edge


class Network(object):
    """
    Una red de vértices conectados por aristas.

    Attributes
    ----------------
        network: graph_tool.Graph
            El grafo que almacena la estructura de la red.
        edge_data: [List[Edge]]
            Lista con la información de las aristas de la red.
        node_map: dict
            Un diccionario que mapea identificadores de nodos a vértices en la red.
        node_layout: LayoutStrategy
            La estrategia de diseño usada para posicionar los nodos al visualizar la red.
        id_to_label: dict
            Diccionario que mapea el identificador de un nodo a su etiqueta.
        community_tree: graph_tool.Graph
            La estructura jerárquica de las comunidades detectadas en la red.
        community_root: int
            El nodo raíz en el árbol de comunidades.
    """
    def __init__(self, graph: graph_tool.Graph):
        """
        Inicializa una nueva instancia de la clase Network.

        Parameters
        ----------
        graph : graph_tool.Graph
            Objeto Graph de graph-tool que representa la red.

        Returns
        -------
        Network
            Una nueva instancia de la clase Network.
        """
        self.network: graph_tool.Graph = graph
        self.edge_data = None

        self.node_map: dict = None
        self.node_layout: LayoutStrategy = None

        self.id_to_label: dict = None
        self.community_tree = None
        self.community_root = None

    @classmethod
    def load(cls, filename: str):
        """
        Carga una red desde un archivo.

        Parameters
        ----------
        filename : str
            Ruta del archivo que contiene la red. Ela rchivo debe estar en formato “gt”, “graphml”, “xml”, “dot” o “gml”.
            Graph_tool recomienda usar "gt" o "graphml" para garantizar la conservación de las propiedades internas del grafo cargado.

        Returns
        -------
        Network
            Instancia de la clase Network creada a partir del archivo.
        """
        if isinstance(filename, Path):
            filename = str(filename)

        network = graph_tool.Graph()
        network.load(filename)

        result = Network(network)
        result.node_map = dict(
            zip(network.vertex_properties["elem_id"], network.vertices())
        )
        result.id_to_label = itemmap(reversed, result.node_map)

        tree_filename = filename + ".community_tree.gt"
        if os.path.exists(tree_filename):
            result.community_tree = graph_tool.Graph()
            result.community_tree.load(tree_filename)
            result.community_root = result.community_tree.graph_properties["root_node"]
            result.communities_per_level = result._build_node_memberships()

        return result

    def save(self, filename: str):
        """
        Guarda la red en un archivo.

        Parameters
        ----------
        filename : str
            Ruta del archivo donde se guardará la red. El archivo debe ser formato "gt", es decir, terminar en ".gt".
        """
        if isinstance(filename, Path):
            filename = str(filename)

        self.network.save(filename, fmt="gt")

        if self.community_tree is not None:
            self.community_tree.save(filename + ".community_tree.gt", fmt="gt")

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
        node_values = sorted(node_values | set(df[target].unique()))
        node_map = dict(zip(node_values, range(len(node_values))))

        df_mapped = df.assign(
            **{
                source_attr: df[source].map(node_map),
                target_attr: df[target].map(node_map),
            }
        )

        network = cls._parse_edgelist(
            df_mapped,
            source_attr,
            target_attr,
            weight_column=weight,
            directed=directed,
        )

        network.vertex_properties["elem_id"] = network.new_vertex_property(
            "object", vals=node_values
        )

        result = cls(network)
        result.node_map = node_map
        result.id_to_label = itemmap(reversed, node_map)
        return result

    @classmethod
    def _parse_edgelist(
        cls,
        df,
        source_column,
        target_column,
        weight_column=None,
        directed=True,
        remove_empty=True,
    ) -> graph_tool.Graph:
        network = graph_tool.Graph(directed=directed)
        n_vertices = max(df[source_column].max(), df[target_column].max()) + 1
        network.add_vertex(n_vertices)

        if weight_column is not None and weight_column in df.columns:
            if remove_empty:
                df = df[df[weight_column] > 0]
            weight_prop = network.new_edge_property("double")
            network.add_edge_list(
                df.assign(**{weight_column: df[weight_column].astype(np.float64)})[
                    [source_column, target_column, weight_column]
                ].values,
                eprops=[weight_prop],
            )
            network.edge_properties["edge_weight"] = weight_prop
            # network.shrink_to_fit()
            return network
        else:
            network.add_edge_list(df[[source_column, target_column]].values)
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

                edge = Edge(src, dst, src_idx, dst_idx, index=i)
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

    @property
    def num_vertices(self):
        return self.network.num_vertices()

    @property
    def num_edges(self):
        return self.network.num_edges()

    @property
    def vertices(self):
        return self.network.vertices()

    @property
    def edges(self):
        return self.network.edges()

    @property
    def is_directed(self):
        return self.network.is_directed()

    @property
    def graph(self):
        return self.network

    def shortest_path(self, src, dst, *args, **kwargs):
        paths = list(
            graph_tool.topology.all_shortest_paths(
                self.network, self.node_map[src], self.node_map[dst], *args, **kwargs
            )
        )

        return [[self.id_to_label[v] for v in path] for path in paths]

    def subgraph(
        self,
        nodes=None,
        vertex_filter=None,
        edge_filter=None,
        keep_positions=True,
        copy=False,
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

        if keep_positions and self.node_layout is not None:
            vertex_positions = [
                self.node_layout.get_position(v_id) for v_id in view.vertices()
            ]
        else:
            vertex_positions = None

        node_map_keys = valfilter(lambda x: x in old_vertex_ids, self.node_map).keys()

        view.purge_vertices()
        view.purge_edges()

        if copy:
            view = graph_tool.Graph(view)
            view.shrink_to_fit()

        result = Network(view)
        result.node_map = dict(zip(node_map_keys, map(int, view.vertices())))
        result.id_to_label = itemmap(reversed, result.node_map)

        if vertex_positions:
            result.layout_nodes(method="precomputed", positions=vertex_positions)
        return result

    @property
    def _edge_weight(self):
        return (
            self.network.edge_properties["edge_weight"]
            if "edge_weight" in self.network.edge_properties
            else None
        )

    def estimate_node_degree(self, degree_type="in"):
        if not degree_type in ("in", "out", "total"):
            raise ValueError("Unsupported node degree")

        vals = getattr(self.network, f"get_{degree_type}_degrees")(
            list(self.network.vertices()),
            eweight=self._edge_weight,
        )

        degree = self.network.new_vertex_property("int", vals=vals)
        self.network.vertex_properties[f"{degree_type}_degree"] = degree

        return degree

    def estimate_betweenness(self, update_nodes=False, update_edges=False):
        node_centrality, edge_centrality = graph_tool.centrality.betweenness(
            self.network
        )

        self.network.edge_properties["betweenness"] = edge_centrality
        self.network.vertex_properties["betweenness"] = node_centrality

        return node_centrality, edge_centrality

    def estimate_pagerank(self, damping=0.85):
        node_centrality = graph_tool.centrality.pagerank(
            self.network, weight=self._edge_weight
        )

        self.network.vertex_properties["pagerank"] = node_centrality
        return node_centrality

    def connected_components(self, directed=True):
        return graph_tool.topology.label_components(self.network, directed=directed)

    def largest_connected_component(self, directed=True, copy=False):
        components = self.connected_components(directed=directed)
        view = self.subgraph(
            vertex_filter=lambda x: components[0][x] == np.argmax(components[1]),
            copy=copy,
        )
        return view

    def detect_communities(
        self,
        random_state=42,
        method="sbm",
        hierarchical_initial_level=1,
        hierarchical_covariate_type="real-exponential",
        ranked=False,
    ):
        np.random.seed(random_state)
        graph_tool.seed_rng(random_state)

        self.communities_per_level = None
        self.community_tree = None
        self.community_root = None

        if method == "sbm":
            self.state = graph_tool.inference.minimize_blockmodel_dl(
                self.network, state_args={"eweight": self._edge_weight}
            )

            self.communities_per_level = None
            self.community_tree = None
            self.community_root = None

            vals = np.array(self.state.get_blocks().a)
        elif method == "hierarchical":
            state_args = dict()

            if self._edge_weight is not None:
                state_args["recs"] = [self._edge_weight]
                state_args["rec_types"] = [hierarchical_covariate_type]

            self.state = graph_tool.inference.minimize_nested_blockmodel_dl(
                self.network, state_args=state_args
            )

            self.community_tree, self.community_root = self._build_community_tree()
            self.communities_per_level = self._build_node_memberships()

            vals = np.array(self.state.get_bs()[hierarchical_initial_level])
        elif method == "ranked":
            state_args = dict()
            state_args["eweight"] = self._edge_weight
            state_args["base_type"] = graph_tool.inference.RankedBlockState
            self.state = graph_tool.inference.minimize_nested_blockmodel_dl(
                self.network, state_args=state_args
            )

            vals = np.array(self.state.levels[0].get_blocks().a)
        else:
            raise ValueError("unsupported method")

        community_prop = self.network.new_vertex_property("int", vals=vals)
        self.network.vertex_properties["community"] = community_prop

    def set_community_level(self, level: int):
        vals = self.get_community_labels(level)
        community_prop = self.network.new_vertex_property("int", vals=vals)
        self.network.vertex_properties["community"] = community_prop

    def get_community_labels(self, level: int = 0):
        if self.communities_per_level is not None:
            return self.communities_per_level[level]

        elif "community" in self.network.vertex_properties:
            return np.array(self.network.vertex_properties["community"].a)

        else:
            raise Exception("must run community detection first")

    def _build_community_tree(self):
        if self.state is None:
            raise Exception("must detect hierarchical communities first.")

        (
            tree,
            membership,
            order,
        ) = graph_tool.inference.nested_blockmodel.get_hierarchy_tree(
            self.state, empty_branches=False
        )
        self.nested_graph = tree

        new_nodes = 0
        root_node_level = 0
        for i, level in enumerate(self.state.get_bs()):
            level = list(level)
            _nodes = len(np.unique(level))
            if _nodes == 1:
                # new_nodes += 1
                break
            else:
                new_nodes += _nodes
                root_node_level += 1

        root_idx = list(tree.vertices())[self.network.num_vertices() + new_nodes]

        tree.graph_properties["root_node"] = tree.new_graph_property(
            "int", val=root_idx
        )

        return tree, root_idx

    def _build_node_memberships(self):
        tree, root = self.community_tree, self.community_root

        depth_edges = graph_tool.search.dfs_iterator(tree, source=root, array=True)

        membership_per_level = defaultdict(lambda: defaultdict(int))

        stack = []
        for src_idx, dst_idx in depth_edges:
            if not stack:
                stack.append(src_idx)

            if dst_idx < self.network.num_vertices():
                # leaf node
                path = [dst_idx]
                path.extend(reversed(stack))

                for level, community_id in enumerate(path):
                    membership_per_level[level][dst_idx] = community_id
            else:
                while src_idx != stack[-1]:
                    # a new community, remove visited branches
                    stack.pop()

                stack.append(dst_idx)

        # removing the lambda enables pickling
        membership_per_level = dict(membership_per_level)
        membership_per_level = valmap(
            lambda x: np.array([x[v_id] for v_id in self.network.vertices()]),
            membership_per_level,
        )

        return membership_per_level
