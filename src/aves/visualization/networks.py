import graph_tool
import graph_tool.topology
import graph_tool.draw
import graph_tool.inference

import numpy as np
import seaborn as sns
import pandas as pd

from aves.features.geometry import bspline
from cytoolz import keyfilter, valfilter, unique, valmap, sliding_window
from collections import defaultdict

from matplotlib.collections import LineCollection
import matplotlib.colors as colors
from matplotlib.patches import FancyArrowPatch, Wedge
from matplotlib.collections import PatchCollection, LineCollection
from sklearn.preprocessing import MinMaxScaler

class NodeLink(object):
    def __init__(self, network, edge_weight=None):
        self.network = network
        self.edge_weight = edge_weight
        
    def estimate_positions(self):
        pass
    
    def plot_edges(self, ax, network=None, color='grey', linewidth=1, alpha=1.0, zorder=0):
        palette = sns.light_palette(color, reverse=True, n_colors=1)
        return self.plot_weighted_edges(ax, network=network, palette=palette, weight_bins=1, alpha=alpha, zorder=zorder)

        
    def plot_weighted_edges(self, ax, network=None, palette='plasma', linewidth=1, alpha=1.0, weight_bins=10, zorder=0, with_arrows=False, min_linewidth=None, min_alpha=None, arrow_shrink=1, arrow_scale=10, log_transform=True):
        if network is None:
            network = self.network
            
        edge_arrow_data = []

        for i, e in enumerate(network.edges()):
            if e.source() == e.target():
                # no support for self connections yet                    
                continue

            arrow = {
                'src_pos': self.node_positions_vector[int(e.source())], 
                'dst_pos': self.node_positions_vector[int(e.target())],
                'weight': self.edge_weight[e] if self.edge_weight is not None else 1
            }

            edge_arrow_data.append(arrow)

        edge_arrow_data = pd.DataFrame.from_records(edge_arrow_data)
        
        if log_transform:
            transform_fn = lambda x: np.log(x + 1)
        else:
            transform_fn = lambda x: x
        
        edge_arrow_data['weight_quantile'], edge_bins = pd.cut(transform_fn(edge_arrow_data.weight), weight_bins, labels=False, retbins=True)
        
        if type(palette) == str:
            edge_colors = sns.color_palette(palette, n_colors=weight_bins)
        else:
            if len(palette) != weight_bins:
                raise ValueError('number of colors is different than number of bins')
            edge_colors = palette
            
        if min_linewidth is None:
            linewidths = np.repeat(linewidth, weight_bins)
        else:
            linewidths = np.linspace(min_linewidth, linewidth, weight_bins)
            
        if min_alpha is None:
            alphas = np.repeat(alpha, weight_bins)
        else:
            alphas = np.linspace(min_alpha, alpha, weight_bins)
            
        for idx, group in edge_arrow_data.groupby('weight_quantile', sort=False):
            arrow_collection = []

            if not network.is_directed() or not with_arrows:
                for i, arrow in enumerate(group.itertuples()):
                    patch = (getattr(arrow, 'src_pos'), getattr(arrow, 'dst_pos'))
                    arrow_collection.append(patch)

                ax.add_collection(LineCollection(arrow_collection, color=edge_colors[idx], linewidth=linewidths[idx], alpha=alphas[idx], zorder=zorder))

            else:
                for i, arrow in enumerate(group.itertuples()):
                    patch = FancyArrowPatch(
                            getattr(arrow, 'src_pos'),
                            getattr(arrow, 'dst_pos'),
                            arrowstyle='-|>',
                            shrinkA=0,
                            shrinkB=arrow_shrink,
                            mutation_scale=arrow_scale,
                            linewidth=linewidths[idx],
                            connectionstyle=None,
                            color=edge_colors[idx],
                            alpha=alphas[idx],
                            linestyle='solid',
                            zorder=zorder
                        )
                    arrow_collection.append(patch)
                    ax.add_patch(patch)

                #ax.add_collection(PatchCollection(arrow_collection, color=edge_colors[idx], linewidth=linewidths[idx], alpha=alphas[idx], zorder=zorder))

    def plot_nodes(self, ax, network=None, size=None, marker='.', color=None, edgecolor=None, palette='colorblind', categories=None, alpha=1.0):
        if network is None:
            pos = self.node_positions_vector
        else:
            idx = list(map(int, network.vertices()))
            pos = self.node_positions_vector[idx]
            
        if categories is not None:
            category_values = np.unique(categories)
            palette = sns.color_palette(palette, n_colors=len(category_values))
            color_map = dict(zip(category_values, palette))
            color = [color_map[c] for c in categories]
            
        ax.scatter(pos[:,0], pos[:,1], s=size, marker=marker, alpha=alpha, color=color, edgecolor=edgecolor)
        
        

class SFDP(NodeLink):
    def __init__(self, network, edge_weight=None):
        super().__init__(network, edge_weight=edge_weight)
        
    def estimate_positions(self, verbose=True):
        self.node_positions = graph_tool.draw.sfdp_layout(self.network, eweight=self.edge_weight, verbose=verbose)
        self.node_positions_vector = np.array(list(self.node_positions))

class ARF(NodeLink):
    def __init__(self, network, edge_weight=None):
        super().__init__(network, edge_weight=edge_weight)
        
    def estimate_positions(self, verbose=True):
        self.node_positions = graph_tool.draw.arf_layout(self.network, weight=self.edge_weight)
        self.node_positions_vector = np.array(list(self.node_positions))

    
        

class HierarchicalEdgeBundling(object):
    def __init__(self, network, edge_weight=None):
        self.network = network
        self.edge_weight = edge_weight
        
    def estimate_blockmodel(self, covariate_type='real-exponential'):
        if self.edge_weight is not None and covariate_type is not None:
            state_args = dict(recs=[self.edge_weight], rec_types=[covariate_type])
            self.state = graph_tool.inference.minimize_nested_blockmodel_dl(self.network, state_args=state_args)
        else:
            self.state = graph_tool.inference.minimize_nested_blockmodel_dl(self.network)
            
        self.block_levels = self.state.get_bs()
      
    def build_community_graph(self):
        self.nested_graph = graph_tool.Graph(directed=False)

        nested_vertex_list = self.nested_graph.add_vertex(self.network.num_vertices())
        self.level_dicts = {}
        self.level_dicts[-1] = dict(zip(range(self.network.num_vertices()), self.nested_graph.vertices()))
        
        self.network_to_nested = dict(zip(self.network.vertices(), self.nested_graph.vertices()))

        for i, level in enumerate(self.block_levels):
            print('level {}: {} nodes'.format(i, len(self.level_dicts[i - 1])))
    
            unique_values = np.unique(level)
            print('unique values:', len(unique_values))
    
            if len(unique_values) > 1:
                group_nodes = self.nested_graph.add_vertex(len(unique_values))
            else:
                group_nodes = [self.nested_graph.add_vertex()]

            level_to_node = dict(zip(unique_values, group_nodes))
            
            for node_id, membership in zip(self.level_dicts[-1].keys(), level):
                self.nested_graph.add_edge(self.level_dicts[i - 1][node_id], level_to_node[membership])
    
            self.level_dicts[i] = level_to_node
                
        print(self.nested_graph.num_vertices(), self.nested_graph.num_edges())
        
        self.radial_positions = graph_tool.draw.radial_tree_layout(self.nested_graph, self.nested_graph.num_vertices() - 1)
        self.radial_positions_vector = np.array(list(self.radial_positions))
        
        
    def set_community_level(self, level=1):
        self.community_level = level
        self.node_to_community = dict(zip(range(self.network.num_vertices()), self.block_levels[0]))

        current_level = 0
        for i in range(1, level):
            update_level = dict(zip(np.unique(self.block_levels[i - 1]), self.block_levels[i]))
            self.node_to_community = valmap(lambda x: update_level[x], self.node_to_community)
            current_level += 1
        
        self.node_to_community = valmap(lambda x: self.level_dicts[current_level][x], self.node_to_community)
        self.community_ids = list(unique(map(int, self.node_to_community.values())))
        
        self.node_angles = np.degrees(np.arctan2(self.radial_positions_vector[:,1], self.radial_positions_vector[:,0]))
        self.node_ratio = np.sqrt(np.dot(self.radial_positions_vector[0], self.radial_positions_vector[0]))
        
        
    def set_community_colors(self, palette='cubehelix', colors=None):
        if colors is None:
            colors = sns.color_palette(palette, n_colors=len(self.community_ids))
        else:
            if len(colors) != len(self.community_ids):
                raise ValueError('Number of colors and communities do not match')
                
        self.community_colors = dict(zip(self.community_ids, colors))
        
        
    def edge_to_spline(self, src, dst, n_points=100, smoothing_factor=0.8):
        if src == dst:
            return None

        vertex_path, edge_path = graph_tool.topology.shortest_path(self.nested_graph, src, dst)
        edge_cp = [self.radial_positions[node_id] for node_id in vertex_path]

        try:    
            smooth_edge = bspline(edge_cp, degree=min(len(edge_cp) - 1, 3), n=n_points)
            source_edge = np.vstack((np.linspace(edge_cp[0][0], edge_cp[-1][0], num=n_points, endpoint=True),
                                     np.linspace(edge_cp[0][1], edge_cp[-1][1], num=n_points, endpoint=True))).T

            if smoothing_factor < 1.0:
                smooth_edge = smooth_edge * smoothing_factor + source_edge * (1.0 - smoothing_factor)

            return smooth_edge
        except ValueError:
            print(src, dst, 'error')
            return None
        
        
    def build_edges(self, n_points=50):
        edges = []
        community_pairs = []
        weights = []
        
        self.n_points = n_points


        for e in self.network.edges():
            if e.source() == e.target():
                continue
                
            src = self.network_to_nested[e.source()]
            dst = self.network_to_nested[e.target()]

            edge = self.edge_to_spline(src, dst, n_points=n_points)
                        
            if edge is not None:
                edges.append(edge)
                community_pairs.append((self.node_to_community[int(src)], self.node_to_community[int(dst)]))
                
                if self.edge_weight is not None:
                    weight = self.edge_weight[e]
                    weights.append(weight)
                else:
                    weights.append(1.0)
                
        self.segments_per_pair = defaultdict(list)

        for edge, pair, weight in zip(edges, community_pairs, weights):
            segments = list(sliding_window(2, edge))
            values = np.linspace(0, 1, num=n_points - 1)

            self.segments_per_pair[pair].append((segments, values, weight))
            
            
    def plot_edges(self, ax, linewidth=1, alpha=0.25, min_linewidth=None, linestyle='solid'):
        if self.edge_weight is not None and min_linewidth is not None:
            width_scaler = MinMaxScaler(feature_range=(min_linewidth, linewidth))
            width_scaler.fit(np.sqrt(self.edge_weight.a).reshape(-1, 1))
        else:
            width_scaler = None
            
        for pair, seg_and_val in self.segments_per_pair.items():
            cmap = colors.LinearSegmentedColormap.from_list("", [self.community_colors[int(pair[0])], self.community_colors[int(pair[1])]])

            all_segments = np.concatenate([s[0] for s in seg_and_val])
            all_values = np.concatenate([s[1] for s in seg_and_val])
            all_weights = np.concatenate([np.repeat(s[2], self.n_points) for s in seg_and_val])
            
            if width_scaler is not None:
                # sin el log hace caput y muestra arte
                linewidths = np.squeeze(width_scaler.transform(np.sqrt(np.array(all_weights)).reshape(-1, 1)))
            else:
                linewidths = linewidth

            edge_collection = LineCollection(all_segments, cmap=cmap, linewidths=linewidths, linestyle=linestyle, alpha=alpha)
            edge_collection.set_array(all_values)
            ax.add_collection(edge_collection)

            
    def plot_nodes(self, ax, size=None, marker='.', alpha=1.0, color=None, edgecolor=None):
        n = self.network.num_vertices()
        ax.scatter(self.radial_positions_vector[:n,0], self.radial_positions_vector[:n,1], s=size, marker=marker, alpha=alpha, color=None, edgecolor=None)
        
        
    def plot_community_wedges(self, ax, wedge_width=0.5, wedge_ratio=None, wedge_offset=0.05, alpha=1.0, fill_gaps=False):
        if wedge_ratio is None:
            wedge_ratio = self.node_ratio + wedge_offset

        wedge_meta = []
        wedge_gap = 180 / self.network.num_vertices() if fill_gaps else 0

        for c_id in self.community_ids:
            
            nodes_in_community = list(valfilter(lambda x: x == c_id, self.node_to_community).keys())

            community_angles = [self.node_angles[n_id] for n_id in nodes_in_community]
            community_angles = [a if a >= 0 else a + 360 for a in community_angles]
            community_angle = self.node_angles[int(c_id)]
            
            if community_angle < 0:
                community_angle += 360
                
            min_angle = min(community_angles)
            max_angle = max(community_angles)
                                    
            extent_angle = max_angle - min_angle
            
            if extent_angle < 0:
                min_angle, max_angle = max_angle, min_angle
            
            if fill_gaps:
                min_angle -= wedge_gap
                max_angle += wedge_gap
            
            wedge_meta.append({'n_nodes': len(nodes_in_community),
                               'center_angle': community_angle,
                               'extent_angle': extent_angle,
                               'min_angle': min_angle,
                               'max_angle': max_angle,
                               'color': self.community_colors[c_id]})
        
                    
        collection = [Wedge(0.0, wedge_ratio + wedge_width, w['min_angle'], w['max_angle'], width=wedge_width) for w in wedge_meta]
        ax.add_collection(PatchCollection(collection, edgecolor='none', 
                                          color=[w['color'] for w in wedge_meta],
                                          alpha=alpha))
        
        
    def n_communities(self):
        return len(self.community_ids)
    
    
    def plot_node_labels(self, ax, labels, ratio=None, offset=0.1):
        if ratio is None:
            ratio = self.node_ratio
            
        ratio += offset
            
        for label, angle_degrees, pos in zip(labels, self.node_angles, self.radial_positions_vector):
            angle = np.radians(angle_degrees)
            va = 'center'

            if pos[0] >= 0:
                ha = 'left'
            else:
                ha = 'right'
            
            text_rotation = angle_degrees

            if text_rotation > 90:
                text_rotation = text_rotation - 180
            elif text_rotation < -90:
                text_rotation = text_rotation + 180

            ax.annotate(label, 
                         (ratio * np.cos(angle), ratio * np.sin(angle)), 
                         rotation=text_rotation, ha=ha, va='center', rotation_mode='anchor',
                         fontsize='small')            
        