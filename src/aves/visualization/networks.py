from aves.visualization.heb import HierarchicalEdgeBundling
from aves.visualization.lines import ColoredCurveCollection
import graph_tool
import graph_tool.topology
import graph_tool.draw
import graph_tool.inference

import numpy as np
import seaborn as sns
import pandas as pd
import geopandas as gpd

import typing

from cytoolz import groupby, pluck, unique
from collections import defaultdict

import matplotlib.colors as colors
from matplotlib.patches import FancyArrowPatch, Wedge
from matplotlib.collections import LineCollection
from sklearn.preprocessing import minmax_scale
from aves.models.network import Network, Edge
from aves.visualization.fdeb import FDB
from collections import defaultdict
from aves.visualization.lines import ColoredCurveCollection
from aves.features.geo import positions_to_array


class NodeLink(object):
    def __init__(self, network: Network):
        self.network = network
        self.bundle_model = None
        self.edge_strategy = None
        self.edge_strategy_args = None
        
        
    def layout_nodes(self, method='sfdp', verbose=True):        
        if method == 'sfdp':
            self.network.set_node_positions(graph_tool.draw.sfdp_layout(self.network.network, eweight=self.network.edge_weight, verbose=verbose))
        elif method == 'arf':        
            self.network.set_node_positions(graph_tool.draw.arf_layout(self.network.network, weight=self.network.edge_weight))
        else:
            raise ValueError('non supported layout')

        if self.edge_strategy is None:
            self.colorize_edges(method='plain')
        else:
            self.edge_strategy.prepare()

    def use_geographical_positions(self, geodf: gpd.GeoDataFrame, node_column=None):
        if len(self.network.node_map) > len(geodf):
            raise ValueError(f'GeoDataFrame has missing vertices')

        if node_column is None:
            geodf = geodf.loc[self.network.node_map.keys()].sort_index()
        else:
            geodf = geodf[geodf[node_column].isin(self.network.node_map.keys())].sort_values(node_column)

        if len(self.network.node_map) != len(geodf):
            raise ValueError(f'Incompatible shapes: {len(self.network.node_map)} nodes and {len(geodf)} shapes. Do you have duplicate rows?')

        node_positions = positions_to_array(geodf.geometry.centroid)
        
        if len(node_positions) != len(self.network.node_map):
            raise ValueError(f'GeoDataFrame and Network have different lengths after filtering nodes')

        self.network.set_node_positions(node_positions)

        if self.edge_strategy is None:
            self.colorize_edges(method='plain')
        else:
            self.edge_strategy.prepare()

        
    def plot_edges(self, ax, *args, **kwargs):
        self.edge_strategy.plot(ax, *args, **kwargs)

        if False:
            edge_arrow_data = pd.Series([e.weight for e in edge_data])

            if log_transform:
                transform_fn = lambda x: np.log(x + 1)
            else:
                transform_fn = lambda x: x
            
            edge_weight_quantile, edge_bins = pd.cut(transform_fn(edge_arrow_data), weight_bins, labels=False, retbins=True)
            
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
                
            for idx, group in groupby(lambda x: x[0], zip(edge_weight_quantile, edge_data)).items():
                # we use pluck because it's a tuple (quantile, arrows)
                arrow_set = list(pluck(1, group))

                polyline_collection = []

                if not network.is_directed() or not with_arrows:
                    arrow_collection = []

                    for arrow in arrow_set:
                        if arrow.polyline is None:
                            patch = (arrow.source, arrow.target)
                            arrow_collection.append(patch)
                        else:
                            polyline_collection.append(arrow.polyline)

                    ax.add_collection(LineCollection(arrow_collection, color=edge_colors[idx], linewidth=linewidths[idx], alpha=alphas[idx], zorder=zorder))
                else:
                    for arrow in arrow_set:
                        if arrow.polyline is None:
                            patch = FancyArrowPatch(
                                    arrow.source,
                                    arrow.target,
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
                            ax.add_patch(patch)
                        else:
                            #TODO: plot these curves with arrows
                            polyline_collection.append(arrow.polyline)
                    #ax.add_collection(PatchCollection(arrow_collection, color=edge_colors[idx], linewidth=linewidths[idx], alpha=alphas[idx], zorder=zorder))
                
                if polyline_collection:
                    edge_collection = LineCollection(polyline_collection, color=edge_colors[idx], linewidth=linewidths[idx], alpha=alphas[idx], zorder=zorder)
                    ax.add_collection(edge_collection)

            return edge_bins

    def plot_nodes(self, ax, min_size=1, size=10, use_weights=None, marker='.', color=None, edgecolor=None, palette='colorblind', categories=None, k=5, alpha=1.0, zorder=0):
        network = self.network

        vertex_idx = list(map(int, network.vertices()))
        pos = network.node_positions_vector

        if use_weights is not None:
            if network.node_weight is None:
                network.weight_nodes()
            degree = network.node_weight

            #print(in_degree)
            size = minmax_scale(degree, feature_range=(min_size, size))
        else:
            size = float(size)
            
        print(type(color))
        if color is None or type(color) == str:
            if categories is None:
                # base case
                # it may resort to the default mpl color
                ax.scatter(pos[:,0], pos[:,1], s=size, marker=marker, alpha=alpha, color=color, edgecolor=edgecolor, zorder=zorder)
            else:
                # we don't care about color because we use the palette attribute
                category_values = list(unique(categories))
                palette = sns.color_palette(palette, n_colors=len(category_values))
                color_map = dict(zip(category_values, palette))
                color = [color_map[c] for c in categories]

                ax.scatter(pos[:,0], pos[:,1], s=size, marker=marker, alpha=alpha, color=color, edgecolor=edgecolor, zorder=zorder)
        elif type(color) in (np.array, np.ndarray):
            if color.shape[0] == 3:
                # it's a single color.
                color = colors.rgb2hex(color)
                ax.scatter(pos[:,0], pos[:,1], s=size, marker=marker, alpha=alpha, color=color, edgecolor=edgecolor, zorder=zorder)
            else:
                # it's an array of values. we interpret it as a list of numbers
                min_value = np.min(color)
                max_value = np.max(color)
                bins = np.linspace(min_value, max_value + (max_value - min_value) * 0.001, num=k + 1)
                color_map = sns.color_palette(palette, n_colors=k)
                color_idx = pd.cut(color, bins=bins, include_lowest=True, labels=False).astype(np.int)
                color = [color_map[c] for c in color_idx]
                ax.scatter(pos[:,0], pos[:,1], s=size, marker=marker, alpha=alpha, c=color, edgecolor=edgecolor, zorder=zorder)
        else:
            raise ValueError('unsupported color parameter')            
            
        return ax

    def plot(self, ax):
        self.plot_edges(ax)
        self.plot_nodes(ax)

    def bundle_edges(self, method: str, *args, **kwargs):
        if method == 'force-directed':
            self.bundle_model = FDB(self.network, *args, **kwargs)
        elif method == 'hierarchical':
            self.bundle_model = HierarchicalEdgeBundling(self.network, *args, **kwargs)
        else:
            raise ValueError(f'method {str} not supported')

        # update node positions
        if self.edge_strategy is not None:
            self.edge_strategy.prepare()

        return self.bundle_model

    def colorize_edges(self, method='weight', **kwargs):
        if method == 'weight':
            self.edge_strategy = WeightStrategy(self.network.edge_data, kwargs.get('k'))
        elif method == 'origin-destination':
            if not self.network.is_directed():
                raise ValueError('method only works with directed graphs')
            self.edge_strategy = ODGradient(self.network.edge_data, kwargs.get('n_points', 30))
        elif method == 'community-gradient':
            if type(self.bundle_model) != HierarchicalEdgeBundling:
                raise ValueError('method only works with HierarchicalEdgeBundling')
            self.edge_strategy = CommunityGradient(self.network.edge_data, kwargs.get('node_communities'))
        elif method == 'attribute':
            raise ValueError(f'{method} is not implemented')
        elif method == 'custom':
            raise ValueError(f'{method} is not implemented')
        elif method == 'plain':
            self.edge_strategy = PlainStrategy(self.network.edge_data, **kwargs)
        else:
            raise ValueError(f'{method} is not supported')

        self.edge_strategy_args = dict(**kwargs)


class EdgeStrategy(object):
    """An interface to methods to render edges with color"""
    def __init__(self, edge_data):
        self.set_data(edge_data)

    def set_data(self, edge_data):
        self.edge_data = edge_data
        self.prepared = False

    def plot(self, ax, *args, **kwargs):
        if not self.prepared:
            self.prepare()

        self.render(ax, *args, **kwargs)

    def get_edge_weights(self):
        return pd.Series([e.weight for e in self.edge_data])

    def prepare(self):
        self.prepare_data()
        self.prepared = True

    def prepare_data(self):
        pass

    def render(self, ax, *args, **kwargs):
        pass

class PlainStrategy(EdgeStrategy):
    """All edges are rendered with the same color"""
    def __init__(self, edge_data, **kwargs):
        super().__init__(edge_data)
        self.lines = None

    def prepare_data(self):
        self.lines = []
        for edge in self.edge_data:
            self.lines.append(edge.points)

        self.lines = np.array(self.lines)

    def render(self, ax, color='#abacab', linewidth=1.0, linestyle='solid', alpha=0.75, zorder=0):
        collection = LineCollection(self.lines, color=color, linewidths=linewidth, linestyle=linestyle, alpha=alpha, zorder=zorder)
        return ax.add_collection(collection)

class WeightStrategy(EdgeStrategy):
    """Colors encode edge weight"""
    def __init__(self, edge_data, k, **kwargs):
        super().__init__(edge_data)
        self.edge_data_per_group = {i: [] for i in range(k)}
        self.strategy_per_group = {i: PlainStrategy(self.edge_data_per_group[i]) for i in range(k)}
        self.k = k
        self.bins = None

    def prepare_data(self):   
        weights = self.get_edge_weights()
        groups, bins = pd.cut(weights, self.k, labels=False, retbins=True)
        self.bins = bins

        for i, edge in zip(groups.values, self.edge_data):
            self.edge_data_per_group[i].append(edge)

        for i in range(self.k):
            self.strategy_per_group[i].set_data(self.edge_data_per_group[i])
            self.strategy_per_group[i].prepare_data()

    def render(self, ax, *args, **kwargs):
        edge_colors = list(reversed(sns.dark_palette(kwargs.pop('color', '#a7a7a7'), n_colors=self.k)))
        print(edge_colors)

        results = []
        for i in range(self.k):
            coll = self.strategy_per_group[i].render(ax, *args, color=edge_colors[i], **kwargs)
            results.append(coll)

        return results

class CommunityGradient(EdgeStrategy):
    """Colors encode communities of each node at the edges"""
    def __init__(self, edge_data, node_communities, **kwargs):
        super().__init__(edge_data)
        self.node_communities = node_communities
        self.community_ids = list(unique(node_communities))
        self.community_links = defaultdict(ColoredCurveCollection)

    def prepare_data(self):
        for edge_data in self.edge_data:
            pair = (self.node_communities[int(edge_data.handle.source())],
                    self.node_communities[int(edge_data.handle.target())])
            
            self.community_links[pair].add_curve(edge_data.points, edge_data.weight)

    def render(self, ax, *args, **kwargs):
        community_colors = dict(zip(self.community_ids, sns.color_palette(kwargs.pop('palette', 'plasma'), n_colors=len(self.community_ids))))
                    
        for pair, colored_lines in self.community_links.items():
            colored_lines.set_colors(source=community_colors[pair[0]], target=community_colors[pair[1]])
            colored_lines.render(ax) 

class ODGradient(EdgeStrategy):
    """Colors encode direction of edges"""
    def __init__(self, edge_data, n_points, **kwargs):
        super().__init__(edge_data)
        self.n_points = n_points
        self.colored_curves = ColoredCurveCollection()

    def prepare_data(self):
        interp = np.linspace(0, 1, num=self.n_points, endpoint=True)

        for edge_data in self.edge_data:
            if type(edge_data.points) == list and len(edge_data.points) == 2:        
                points = np.array([(edge_data.source * (1 - t) + edge_data.target * t) for t in interp])
            elif type(edge_data.points) == np.array and edge_data.points.shape[0] == 2:
                points = np.array([(edge_data.source * (1 - t) + edge_data.target * t) for t in interp])
            else:
                points = edge_data.points
            
            self.colored_curves.add_curve(points, edge_data.weight)

    def render(self, ax, *args, **kwargs):
        self.colored_curves.set_colors(source=kwargs.get('source_color', 'blue'), target=kwargs.get('target_color', 'red'))
        self.colored_curves.render(ax) 
