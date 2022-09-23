from collections import defaultdict

import graph_tool
import graph_tool.draw
import graph_tool.inference
import graph_tool.search
import graph_tool.topology
import numpy as np
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge

from aves.features.geometry import bspline
from aves.models.network import Network


class HierarchicalEdgeBundling(object):
    def __init__(
        self,
        network: Network,
        hierarchy_tree: graph_tool.Graph,
        root_idx: int = 0,
        points_per_edge=50,
        path_smoothing_factor=0.8,
    ):
        self.network = network

        self.nested_graph = hierarchy_tree
        self.root_idx = root_idx

        self.root_dist_map, self.root_pred_map = graph_tool.topology.shortest_distance(
            self.nested_graph, source=self.root_idx, pred_map=True
        )

        self.build_structure()
        self.build_edges(
            n_points=points_per_edge, smoothing_factor=path_smoothing_factor
        )

    def build_structure(self):
        from aves.visualization.networks import NodeLink

        self.radial_positions = np.array(
            list(
                graph_tool.draw.radial_tree_layout(
                    self.nested_graph,
                    self.nested_graph.vertex(self.root_idx),
                )
            )
        )

        self.node_to_radial_idx = dict(
            zip(self.nested_graph.vertices(), range(self.nested_graph.num_vertices()))
        )

        self.node_angles = np.degrees(
            np.arctan2(self.radial_positions[:, 1], self.radial_positions[:, 0])
        )
        self.node_angles_dict = dict(
            zip(map(int, self.nested_graph.vertices()), self.node_angles)
        )
        self.node_ratio = np.sqrt(
            np.dot(self.radial_positions[0], self.radial_positions[0])
        )

        self.network.layout_nodes(
            method="precomputed",
            positions=self.radial_positions[: self.network.num_vertices],
            angles=self.node_angles,
            ratios=np.sqrt(
                np.sum(self.radial_positions * self.radial_positions, axis=1)
            ),
        )

        self.community_graph = Network(
            graph_tool.GraphView(
                self.nested_graph,
                directed=True,
                vfilt=lambda x: x >= self.network.num_vertices,
            )
        )
        self.community_nodelink = NodeLink(self.community_graph)
        self.community_nodelink.layout_nodes(
            method="precomputed",
            positions=self.radial_positions[self.network.num_vertices :],
            angles=self.node_angles,
            ratios=self.node_ratio,
        )
        self.community_nodelink.set_node_drawing(method="plain")
        self.community_nodelink.set_edge_drawing(method="plain")

    def edge_to_spline(self, control_points, n_points, smoothing_factor):
        try:
            smooth_edge = bspline(
                control_points, degree=min(len(control_points) - 1, 3), n=n_points
            )
            source_edge = np.vstack(
                (
                    np.linspace(
                        control_points[0][0],
                        control_points[-1][0],
                        num=n_points,
                        endpoint=True,
                    ),
                    np.linspace(
                        control_points[0][1],
                        control_points[-1][1],
                        num=n_points,
                        endpoint=True,
                    ),
                )
            ).T

            if smoothing_factor < 1.0:
                smooth_edge = smooth_edge * smoothing_factor + source_edge * (
                    1.0 - smoothing_factor
                )

            return smooth_edge
        except ValueError as ex:
            raise ValueError(f"Could not build a spline in {control_points}. {ex}")

    def build_edges(self, n_points=50, smoothing_factor=0.8):
        edge_ids_per_source = defaultdict(list)
        built_edges = dict()

        for e in self.network.edge_data:
            src = e.index_pair[0]
            dst = e.index_pair[1]

            if src == dst:
                raise Exception(
                    "Self-pointing edges are not supported ({src} -> {dst})"
                )

            edge_ids_per_source[src].append(dst)

        self.nested_graph.set_directed(False)
        for src, dst_nodes in edge_ids_per_source.items():
            _, pred_map = graph_tool.topology.shortest_distance(
                self.nested_graph, src, dst_nodes, pred_map=True
            )

            for dst in dst_nodes:
                vertex_path, _ = graph_tool.topology.shortest_path(
                    self.nested_graph, src, dst, pred_map=pred_map
                )

                edge_cp = [
                    self.radial_positions[self.node_to_radial_idx[node_id]]
                    for node_id in vertex_path
                ]
                built_edges[(src, dst)] = self.edge_to_spline(
                    edge_cp, n_points, smoothing_factor
                )

        self.nested_graph.set_directed(True)

        for e in self.network.edge_data:
            curve = built_edges[(e.index_pair[0], e.index_pair[1])]

            if curve is not None:
                e.points = curve

    def plot_community_wedges(
        self,
        ax,
        level=1,
        wedge_width=0.5,
        wedge_ratio=None,
        wedge_offset=0.05,
        wedge_kwargs=None,
        alpha=1.0,
        fill_gaps=False,
        palette="plasma",
        label_func=None,
        label_kwargs=None,
    ):

        if wedge_ratio is None:
            wedge_ratio = self.node_ratio + wedge_offset

        nodes = np.array(list(map(int, self.network.vertices)))
        community_ids = sorted(set(self.network.communities_per_level[level]))

        if isinstance(palette, dict):
            if len(palette) != len(community_ids):
                raise ValueError(
                    "the number of colors does not match the number of categories"
                )
            if set(palette.keys()) != set(community_ids):
                raise ValueError(
                    "the provided palette does not contain all community ids"
                )

            community_colors = palette
        else:
            if isinstance(palette, str):
                palette = sns.color_palette(palette, n_colors=len(community_ids))
            elif palette is not None:
                # assume it's an iterable of colors
                palette = list(palette)
                if len(palette) != len(community_ids):
                    raise ValueError(
                        "the number of colors does not match the number of categories"
                    )
            else:
                raise ValueError(
                    "palette must be a valid name or an iterable of colors"
                )

            community_colors = dict(zip(community_ids, palette))

        wedge_meta = []
        wedge_gap = 180 / self.network.num_vertices if fill_gaps else 0

        # fom https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_and_donut_labels.html
        bbox_props = dict(boxstyle="square,pad=0.3", fc="none", ec="none")

        kw = dict(
            arrowprops=dict(arrowstyle="-", color="#abacab"),
            bbox=bbox_props,
            zorder=0,
            va="center",
            fontsize=8,
        )
        if label_kwargs is not None:
            kw.update(label_kwargs)

        for c_id in community_ids:

            nodes_in_community = nodes[
                self.network.communities_per_level[level] == c_id
            ]

            community_angles = [
                self.node_angles_dict[n_id] for n_id in nodes_in_community
            ]
            community_angles = [a if a >= 0 else a + 360 for a in community_angles]
            community_angle = self.node_angles_dict[int(c_id)]

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

            wedge_meta.append(
                {
                    "community_id": c_id,
                    "n_nodes": len(nodes_in_community),
                    "center_angle": community_angle,
                    "extent_angle": extent_angle,
                    "min_angle": min_angle,
                    "max_angle": max_angle,
                    "color": community_colors[c_id],
                }
            )

            if label_func is not None:
                community_label = label_func(c_id)
                if community_label:
                    ratio = wedge_ratio + wedge_width

                    mid_angle = 0.5 * (max_angle + min_angle)
                    mid_angle_radians = np.radians(mid_angle)

                    pos_x, pos_y = ratio * np.cos(mid_angle_radians), ratio * np.sin(
                        mid_angle_radians
                    )

                    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(pos_x))]
                    connectionstyle = "angle,angleA=0,angleB={}".format(mid_angle)
                    kw["arrowprops"].update({"connectionstyle": connectionstyle})
                    ax.annotate(
                        community_label,
                        xy=(pos_x, pos_y),
                        xytext=(1.15 * pos_x, 1.25 * pos_y),
                        horizontalalignment=horizontalalignment,
                        **kw,
                    )

        collection = [
            Wedge(
                0.0,
                wedge_ratio + wedge_width,
                w["min_angle"],
                w["max_angle"],
                width=wedge_width,
            )
            for w in wedge_meta
        ]

        collection_args = dict(edgecolor="none", alpha=alpha)
        if wedge_kwargs is not None:
            collection_args.update(wedge_kwargs)

        ax.add_collection(
            PatchCollection(
                collection, color=[w["color"] for w in wedge_meta], **collection_args
            )
        )

        return wedge_meta, collection

    def plot_community_labels(self, ax, level=None, ratio=None, offset=0.05):
        if ratio is None:
            ratio = self.node_ratio + offset

        nodes = np.array(list(map(int, self.network.vertices)))
        community_ids = sorted(set(self.network.communities_per_level[level]))

        for c_id in community_ids:
            nodes_in_community = nodes[
                self.network.communities_per_level[level] == c_id
            ]

            community_angles = [
                self.node_angles_dict[n_id] for n_id in nodes_in_community
            ]
            community_angles = [a if a >= 0 else a + 360 for a in community_angles]
            community_angle = self.node_angles[int(c_id)]

            if community_angle < 0:
                community_angle += 360

            min_angle = min(community_angles)
            max_angle = max(community_angles)

            mid_angle = 0.5 * (max_angle + min_angle)
            mid_angle_radians = np.radians(mid_angle)

            pos_x, pos_y = ratio * np.cos(mid_angle_radians), ratio * np.sin(
                mid_angle_radians
            )

            ha = "left" if pos_x >= 0 else "right"

            if mid_angle > 90:
                mid_angle = mid_angle - 180
            elif mid_angle < -90:
                mid_angle = mid_angle + 180

            ax.annotate(
                f"{c_id}",
                (pos_x, pos_y),
                rotation=mid_angle,
                ha=ha,
                va="center",
                rotation_mode="anchor",
                fontsize="small",
            )

    def plot_community_network(self, ax):
        self.community_nodelink.plot_nodes(ax, color="blue", marker="s")
        self.community_nodelink.plot_edges(ax, color="black", linewidth=2, alpha=0.8)
