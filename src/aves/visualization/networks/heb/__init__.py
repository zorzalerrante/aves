import typing
from collections import defaultdict

import graph_tool
import graph_tool.draw
import graph_tool.inference
import graph_tool.search
import graph_tool.topology
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import seaborn as sns
from cytoolz import valfilter, valmap
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Wedge

from aves.features.geometry import bspline
from aves.models.network import Network


class HierarchicalEdgeBundling(object):
    def __init__(
        self,
        network: Network,
        state=None,
        covariate_type=None,
        points_per_edge=50,
        path_smoothing_factor=0.8,
    ):
        self.network = network
        self.state = state
        if state is not None:
            self.block_levels = self.state.get_bs()
        else:
            self.estimate_blockmodel(covariate_type=covariate_type)

        self.build_community_graph()
        self.build_node_memberships()
        self.build_edges(
            n_points=points_per_edge, smoothing_factor=path_smoothing_factor
        )

    def estimate_blockmodel(self, covariate_type="real-exponential"):
        if self.network.edge_weight is not None and covariate_type is not None:
            state_args = dict(
                recs=[self.network.edge_weight], rec_types=[covariate_type]
            )
            self.state = graph_tool.inference.minimize_nested_blockmodel_dl(
                self.network.graph(), state_args=state_args
            )
        else:
            self.state = graph_tool.inference.minimize_nested_blockmodel_dl(
                self.network.graph()
            )

        self.block_levels = self.state.get_bs()

    def get_node_memberships(self, level):
        return [
            self.membership_per_level[level][int(node_id)]
            for node_id in self.network.vertices()
        ]

    def build_community_graph(self):
        from aves.visualization.networks import NodeLink

        (
            tree,
            membership,
            order,
        ) = graph_tool.inference.nested_blockmodel.get_hierarchy_tree(
            self.state, empty_branches=False
        )
        self.nested_graph = tree
        self.nested_graph.set_directed(False)

        self.radial_positions = np.array(
            list(
                graph_tool.draw.radial_tree_layout(
                    self.nested_graph, self.nested_graph.num_vertices() - 1
                )
            )
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
            positions=self.radial_positions[: self.network.num_vertices()],
            angles=self.node_angles,
            ratios=np.sqrt(
                np.sum(self.radial_positions * self.radial_positions, axis=1)
            ),
        )

        self.community_graph = Network(
            graph_tool.GraphView(
                self.nested_graph,
                directed=True,
                vfilt=lambda x: x >= self.network.num_vertices(),
            )
        )
        self.community_nodelink = NodeLink(self.community_graph)
        self.community_nodelink.layout_nodes(
            method="precomputed",
            positions=self.radial_positions[self.network.num_vertices() :],
            angles=self.node_angles,
            ratios=self.node_ratio,
        )
        self.community_nodelink.set_node_drawing(method="plain")
        self.community_nodelink.set_edge_drawing(method="plain")

    def build_node_memberships(self):
        self.nested_graph.set_directed(True)

        depth_edges = graph_tool.search.dfs_iterator(
            self.nested_graph, source=self.nested_graph.num_vertices() - 1, array=True
        )

        self.membership_per_level = defaultdict(lambda: defaultdict(int))

        stack = []
        for src_idx, dst_idx in depth_edges:
            if not stack:
                stack.append(src_idx)

            if dst_idx < self.network.num_vertices():
                # leaf node
                path = [dst_idx]
                path.extend(reversed(stack))

                for level, community_id in enumerate(path):
                    self.membership_per_level[level][dst_idx] = community_id
            else:
                while src_idx != stack[-1]:
                    # a new community, remove visited branches
                    stack.pop()

                stack.append(dst_idx)

        self.nested_graph.set_directed(False)

        # removing the lambda enables pickling
        self.membership_per_level = dict(self.membership_per_level)
        self.membership_per_level = valmap(dict, self.membership_per_level)

    def edge_to_spline(self, src, dst, n_points, smoothing_factor):
        if src == dst:
            raise Exception("Self-pointing edges are not supported")

        vertex_path, edge_path = graph_tool.topology.shortest_path(
            self.nested_graph, src, dst
        )
        edge_cp = [self.radial_positions[int(node_id)] for node_id in vertex_path]

        try:
            smooth_edge = bspline(edge_cp, degree=min(len(edge_cp) - 1, 3), n=n_points)
            source_edge = np.vstack(
                (
                    np.linspace(
                        edge_cp[0][0], edge_cp[-1][0], num=n_points, endpoint=True
                    ),
                    np.linspace(
                        edge_cp[0][1], edge_cp[-1][1], num=n_points, endpoint=True
                    ),
                )
            ).T

            if smoothing_factor < 1.0:
                smooth_edge = smooth_edge * smoothing_factor + source_edge * (
                    1.0 - smoothing_factor
                )

            return smooth_edge
        except ValueError:
            print(src, dst, "error")
            return None

    def build_edges(self, n_points=50, smoothing_factor=0.8):
        for e in self.network.edge_data:
            src = e.index_pair[0]
            dst = e.index_pair[1]

            curve = self.edge_to_spline(src, dst, n_points, smoothing_factor)

            if curve is not None:
                e.points = curve

    def plot_community_wedges(
        self,
        ax,
        level=1,
        wedge_width=0.5,
        wedge_ratio=None,
        wedge_offset=0.05,
        alpha=1.0,
        fill_gaps=False,
        palette="plasma",
        label_func=None,
    ):

        if wedge_ratio is None:
            wedge_ratio = self.node_ratio + wedge_offset

        community_ids = sorted(set(self.membership_per_level[level].values()))
        community_colors = dict(
            zip(community_ids, sns.color_palette(palette, n_colors=len(community_ids)))
        )

        wedge_meta = []
        wedge_gap = 180 / self.network.num_vertices() if fill_gaps else 0

        # fom https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_and_donut_labels.html
        bbox_props = dict(boxstyle="square,pad=0.3", fc="none", ec="none")
        kw = dict(
            arrowprops=dict(arrowstyle="-", color="#abacab"),
            bbox=bbox_props,
            zorder=0,
            va="center",
            fontsize=8,
        )

        for c_id in community_ids:

            nodes_in_community = list(
                valfilter(lambda x: x == c_id, self.membership_per_level[level]).keys()
            )

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
                        xytext=(1.35 * pos_x, 1.4 * pos_y),
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
        ax.add_collection(
            PatchCollection(
                collection,
                edgecolor="none",
                color=[w["color"] for w in wedge_meta],
                alpha=alpha,
            )
        )

        return wedge_meta, collection

    def plot_community_labels(self, ax, level=None, ratio=None, offset=0.05):
        if ratio is None:
            ratio = self.node_ratio + offset

        community_ids = set(self.membership_per_level[level].values())

        for c_id in community_ids:

            nodes_in_community = list(
                valfilter(lambda x: x == c_id, self.membership_per_level[level]).keys()
            )

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
