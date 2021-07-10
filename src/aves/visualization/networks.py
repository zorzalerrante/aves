import graph_tool
import graph_tool.topology
import graph_tool.draw
import graph_tool.inference

import numpy as np
import seaborn as sns
import pandas as pd

import typing

from aves.features.network import graph_from_pandas_edgelist
from aves.features.geometry import bspline
from cytoolz import keyfilter, valfilter, unique, valmap, sliding_window, groupby, pluck
from collections import defaultdict

import matplotlib.colors as colors
from matplotlib.patches import FancyArrowPatch, Wedge
from matplotlib.collections import PatchCollection, LineCollection
from sklearn.preprocessing import MinMaxScaler, minmax_scale




class NodeLink(object):
    def __init__(self, network):
        self.network = network
        
    def layout_nodes(self, method='sfdp', verbose=True):        
        if method == 'sfdp':
            self.network.set_node_positions(graph_tool.draw.sfdp_layout(self.network.network, eweight=self.network.edge_weight, verbose=verbose))
        elif method == 'arf':        
            self.network.set_node_positions(graph_tool.draw.arf_layout(self.network.network, weight=self.network.edge_weight))
        else:
            raise ValueError('non supported layout')

    def plot_edges(self, ax, color='grey', linewidth=1, alpha=1.0, zorder=0, network: typing.Optional[graph_tool.GraphView]=None, with_arrows=False):
        palette = sns.light_palette(color, reverse=True, n_colors=1)
        return self.plot_weighted_edges(ax, palette=palette, weight_bins=1, alpha=alpha, zorder=zorder, network=network, with_arrows=with_arrows)

        
    def plot_weighted_edges(self, ax, palette='plasma', linewidth=1, alpha=1.0, weight_bins=10, zorder=1, with_arrows=False, min_linewidth=None, min_alpha=None, arrow_shrink=1, arrow_scale=10, log_transform=True, network: typing.Optional[graph_tool.GraphView]=None):
        if network is None:
            network = self.network
            edge_data = self.network.edge_data
        else:
            edge_pairs = set((int(e.source()), int(e.target())) for e in network.edges())
            edge_data = [e for e in self.network.edge_data if e.index_pair in edge_pairs]
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

    def plot_nodes(self, ax, min_size=1, size=10, use_weights=None, marker='.', color=None, edgecolor=None, palette='colorblind', categories=None, k=5, alpha=1.0, zorder=0, network: typing.Optional[graph_tool.GraphView]=None):
        #TODO: consider the view
        if network is None:
            network = self.network

        vertex_idx = list(map(int, network.vertices()))
        pos = np.array([network.node_position(idx) for idx in vertex_idx])

        if use_weights == 'in-degree':
            in_degree = network.get_in_degrees(vertex_idx)
            #print(in_degree)
            size = minmax_scale(in_degree, feature_range=(min_size, size))
        elif use_weights == 'out-degree':
            pass
        elif use_weights == 'total-degree':
            pass
        elif use_weights is None:
            size = float(size)
        else:
            raise ValueError('use_weights must be none or in/out/total-degree')
            
        print(type(color))
        if color is None or type(color) == str:
            if categories is None:
                # base case
                # it may resort to the default mpl color
                ax.scatter(pos[:,0], pos[:,1], s=size, marker=marker, alpha=alpha, color=color, edgecolor=edgecolor, zorder=zorder)
            else:
                # we don't care about color because we use the palette attribute
                category_values = np.unique(categories)
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


        
        