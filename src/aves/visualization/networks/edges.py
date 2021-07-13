import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.collections import LineCollection
from aves.visualization.lines import ColoredCurveCollection
from cytoolz import unique
from collections import defaultdict

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
