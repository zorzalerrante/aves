import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
import numpy as np
import seaborn as sns
from cytoolz import unique
from sklearn.preprocessing import minmax_scale

from aves.models.network import Network
from aves.visualization.primitives import RenderStrategy


class NodeStrategy(RenderStrategy):
    """An interface to methods to render nodes"""

    def __init__(self, network: Network, **kwargs):
        self.network = network
        super().__init__(network.node_layout.node_positions_vector)

    def prepare_data(self):
        pass

    def name(self):
        return "node-strategy"


class PlainNodes(NodeStrategy):
    """All nodes are rendered as a scatterplot"""

    def __init__(self, network: Network, **kwargs):
        super().__init__(network, **kwargs)

        weights = kwargs.get("weights", None)

        if weights is not None and not type(weights) in (np.array, np.ndarray):
            raise ValueError(f"weights must be np.array instead of {type(weights)}.")

        self.weights: np.array = weights
        self.size: np.array = None

        self.node_categories = kwargs.pop("categories", None)
        if self.node_categories is not None:
            self.unique_categories = sorted(unique(self.node_categories))

    def prepare_data(self):
        if self.weights is not None:
            self.size = minmax_scale(np.sqrt(self.weights), feature_range=(0.01, 1.0))

    def render(self, ax, *args, **kwargs):
        node_size = kwargs.pop("node_size", 10)

        if self.node_categories:
            palette_name = kwargs.pop("palette", "plasma")
            if isinstance(palette_name, str):
                palette = sns.color_palette(
                    palette_name, n_colors=len(self.unique_categories)
                )
            elif palette_name is not None:
                # assume it's an iterable of colors
                palette = list(palette_name)
            else:
                raise ValueError(
                    "palette must be a valid name or an iterable of colors"
                )
            color_map = dict(zip(self.unique_categories, palette))
            c = [color_map[c] for c in self.node_categories]
        else:
            c = None

        if self.weights is None:
            ax.scatter(
                self.data[:, 0], self.data[:, 1], *args, s=node_size, c=c, **kwargs
            )
        else:
            ax.scatter(
                self.data[:, 0],
                self.data[:, 1],
                *args,
                s=self.size * node_size,
                c=c,
                **kwargs,
            )

    def name(self):
        return "plain"


class LabeledNodes(PlainNodes):
    def __init__(self, network: Network, label_property, **kwargs):
        super().__init__(network, **kwargs)
        self.labels = []
        self.label_property = label_property

        self.radial = kwargs.get("radial", False)
        self.offset = kwargs.get("offset", 0.0)

    def prepare_data(self):
        super().prepare_data()

        graph = self.network.graph()

        for idx in graph.vertices():
            label = self.label_property[idx]
            if label:
                text_args = {}

                if self.radial:
                    degrees = self.network.node_layout.get_angle(idx)
                    ratio = self.network.node_layout.get_ratio(idx) + self.offset
                    radians = np.radians(degrees)

                    pos = np.array([ratio * np.cos(radians), ratio * np.sin(radians)])

                    text_args["ha"] = "left" if pos[0] >= 0 else "right"
                    text_rotation = degrees
                    if text_rotation > 90:
                        text_rotation = text_rotation - 180
                    elif text_rotation < -90:
                        text_rotation = text_rotation + 180

                    text_args["rotation"] = text_rotation

                    text_args["rotation_mode"] = "anchor"
                else:
                    pos = self.data[int(idx)]
                    text_args["ha"] = "center"

                text_args["va"] = "center"

                self.labels.append((pos, label, text_args))

    def render(self, ax, *args, **kwargs):
        fontsize = kwargs.pop("fontsize", "medium")
        text_color = kwargs.pop("text_color", "white")

        super().render(ax, *args, **kwargs)

        for pos, label, text_args in self.labels:
            text_args["fontsize"] = fontsize
            text_args["color"] = text_color

            text = ax.text(pos[0], pos[1], label, **text_args)
            text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=1, foreground="black"),
                    path_effects.Normal(),
                ]
            )

    def name(self):
        return "labeled"
