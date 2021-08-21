from collections import defaultdict
from itertools import repeat

import matplotlib.colors as colors
import numpy as np
from cytoolz import sliding_window
from matplotlib.collections import LineCollection
from sklearn.preprocessing import minmax_scale


class ColoredCurveCollection(object):
    def __init__(
        self, source_color=None, target_color=None, linewidth=1.0, min_linewidth=None
    ):
        self.curves_per_length = defaultdict(list)
        self.prepared = False
        self.source_color = source_color
        self.target_color = target_color
        self.colormap = None
        self.segments_per_length = None
        self.values_per_length = None
        self.linewidth = None

    def render(self, ax, *args, **kwargs):
        if not self.prepared:
            self.prepare()

        results = []

        for group in self.prepared_data.values():
            collection = LineCollection(
                group["segments"],
                *args,
                cmap=self.colormap,
                linewidths=group["linewidth"],
                **kwargs
            )
            collection.set_array(group["color_values"])
            result = ax.add_collection(collection)
            results.append(result)

        return results

    def add_curve(self, curve: np.array, value: float = 1.0):
        length = curve.shape[0]
        self.curves_per_length[length].append((curve, value))
        self.prepared = False

    def add_curves(self, curves, values):
        for c, v in zip(curves, values):
            self.add_curve(c, v)

    def set_colors(self, source=None, target=None):
        if source is not None:
            self.source_color = source
        if target is not None:
            self.target_color = target

        self.colormap = colors.LinearSegmentedColormap.from_list(
            "", [self.source_color, self.target_color]
        )

    def set_linewidth(self, linewidth=1.0, min_linewidth=None, transform=minmax_scale):
        self.linewidth = linewidth
        self.min_linewidth = min_linewidth
        self.linewidth_transform = transform
        self.prepared = False

    def prepare(self):
        if self.colormap is None:
            self.colormap = colors.LinearSegmentedColormap.from_list(
                "", [self.source_color, self.target_color]
            )

        if self.linewidth is None:
            self.set_linewidth()

        self.prepared_data = {}

        for n_points, pairs in self.curves_per_length.items():
            segments = np.concatenate([list(sliding_window(2, p[0])) for p in pairs])
            weights = np.concatenate([list(repeat(p[1], n_points)) for p in pairs])
            color_values = np.concatenate(
                list(repeat(np.linspace(0, 1, num=n_points - 1), len(pairs)))
            )

            if self.min_linewidth is not None:
                linewidth = np.squeeze(
                    self.linewidth_transform(
                        weights.reshape(-1, 1),
                        feature_range=(self.min_linewidth, self.linewidth),
                    )
                )
            else:
                linewidth = self.linewidth

            self.prepared_data[n_points] = {
                "segments": segments,
                "weights": weights,
                "color_values": color_values,
                "linewidth": linewidth,
            }

        self.prepared = True
