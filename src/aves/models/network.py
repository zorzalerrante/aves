import graph_tool
import graph_tool.topology
import graph_tool.inference
import numpy as np
import pandas as pd

EPS = 1e-6

class Edge(object):
    def __init__(self, source, target, source_idx, target_idx, weight=None, index=-1, handle=None):
        self.source = source
        self.target = target
        self.handle = handle

        self._vector = self.target - self.source 

        if np.allclose(self.source, self.target, atol=EPS):
            self._length = EPS
        else:
            self._length = np.sqrt(np.dot(self._vector, self._vector))

        self._unit_vector = self._vector / self._length
        self._mid_point = (self.source + self.target) * 0.5

        if weight is None:
            self.weight = 1
        else:
            self.weight = weight

        self.index = index
        # this is filled by external algorithms
        self.polyline = None
        self.index_pair = (source_idx, target_idx)


    def as_vector(self):
        return self._vector

    def length(self):
        return self._length

    def project(self, point):
        L = self._length
        p_vec = point - self.source
        return self.source + np.dot(p_vec, self._unit_vector)  * self._unit_vector

class Network(object):
    def __init__(self, graph: graph_tool.Graph, edge_weight=None):
        self.network = graph
        self.edge_weight = edge_weight
        self.edge_data = None
        self.node_positions = None
        self.node_positions_dict = None
        self.node_positions_vector = None
        self.node_map = None
        
    @classmethod
    def from_edgelist(cls, df: pd.DataFrame, source='source', target='target', directed=True, weight=None, map_nodes=True):
        if map_nodes:
            source_attr = f'{source}__mapped__'
            target_attr = f'{target}__mapped__'

            node_values = set(df[source].unique())
            node_values = node_values | set(df[target].unique())
            node_map = dict(zip(sorted(node_values), range(len(node_values))))

            df_mapped = df.assign(**{
                source_attr: df[source].map(node_map),
                target_attr: df[target].map(node_map)
            })
        else:
            source_attr = source
            target_attr = target
            df_mapped = df
            node_map = None

        if weight is not None:
            network, edge_weight = cls._parse_edgelist(df_mapped, source=source_attr, target=target_attr, weight=weight, directed=directed)
        else:
            network = cls._parse_edgelist(df_mapped, source=source_attr, target=target_attr, weight=None, directed=directed)
            edge_weight = None
        result = cls(network, edge_weight=edge_weight)
        result.node_map = node_map
        return result

    @classmethod
    def _parse_edgelist(cls, df, source='source', target='target', weight='weight', directed=True, remove_empty=True):
        network = graph_tool.Graph(directed=directed)
        n_vertices = max(df[source].max(), df[target].max()) + 1
        vertex_list = network.add_vertex(n_vertices)
        
        if weight is not None and weight in df.columns:
            if remove_empty:
                df = df[df[weight] > 0]
            weight_prop = network.new_edge_property('double')
            network.add_edge_list(df.assign(**{weight: df[weight].astype(np.float64)})[[source, target, weight]].values, eprops=[weight_prop])
            #network.shrink_to_fit()
            return network, weight_prop
        else:
            network.add_edge_list(df[[source, target]].values)
            #network.shrink_to_fit()
            return network

    def build_edge_data(self):
        self.edge_data = []

        for i, e in enumerate(self.network.edges()):
            src_idx = int(e.source())
            dst_idx = int(e.target())
            if src_idx == dst_idx:
                # no support for self connections yet                    
                continue

            src = self.node_positions_dict[src_idx]
            dst = self.node_positions_dict[dst_idx]
            weight = self.edge_weight[e] if self.edge_weight is not None else 1

            edge = Edge(src, dst, src_idx, dst_idx, weight=weight, index=i, handle=e)
            self.edge_data.append(edge)
    

    def set_node_positions(self, layout_vector):
        '''
        @param layout_vector outcome from a layout method from graphtool or an array with positions, in vertex order
        '''

        if type(layout_vector) in (pd.DataFrame, pd.Series):
            layout_vector = layout_vector.values

        self.node_positions = layout_vector
        self.node_positions_vector = np.array(list(layout_vector))
        self.node_positions_dict = dict(zip(list(map(int, self.network.vertices())), list(self.node_positions_vector)))
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

    def node_position(self, node_idx):
        return self.node_positions_dict[node_idx]

    def graph(self):
        return self.network

    def shortest_path(self, src, dst):
        return list(graph_tool.topology.all_shortest_paths(self.network, src, dst))

    def view(self, vertex_filter=None, edge_filter=None):
        result = Network(graph_tool.GraphView(self.network, vfilt=vertex_filter, efilt=edge_filter))
        vertex_positions = [self.node_positions[v_id] for v_id in result.network.vertices()]
        result.set_node_positions(vertex_positions)
        return result