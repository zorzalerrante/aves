from abc import ABC, abstractmethod

import geopandas as gpd
import graph_tool
import graph_tool.draw
import graph_tool.topology
import numpy as np

from aves.features.geo import positions_to_array

from .base import Network


class LayoutStrategy(ABC):
    def __init__(self, network: Network, name: str):
        self.network = network
        self.name = name
        self.node_positions = None
        self.node_positions_dict: dict = None
        self.node_positions_vector: np.array = None

    @abstractmethod
    def layout(self):
        pass

    def _post_layout(self):
        pass

    def layout_nodes(self, *args, **kwargs):
        self.layout(*args, **kwargs)

        self.node_positions_vector = np.array(list(self.node_positions))
        self.node_positions_dict = dict(
            zip(
                list(map(int, self.network.vertices())),
                list(self.node_positions_vector),
            )
        )

        self._post_layout()

        return self.node_positions

    def get_position(self, idx):
        idx = int(idx)
        return self.node_positions_dict[idx]

    def get_angle(self, idx):
        raise NotImplementedError("this class doesn't work with angles")

    def get_ratio(self, idx):
        raise NotImplementedError("this class doesn't work with ratios")

    def positions(self):
        return self.node_positions_vector


class ForceDirectedLayout(LayoutStrategy):
    def __init__(self, network: Network):
        super().__init__(network, "force-directed")

    def layout(self, *args, **kwargs):
        method = kwargs.pop("algorithm", "sfdp")

        if not method in ("sfdp", "arf"):
            raise ValueError(f"unsupported method: {method}")

        if method == "sfdp":
            self.node_positions = graph_tool.draw.sfdp_layout(
                self.network.graph(),
                eweight=self.network.edge_weight,
                verbose=kwargs.pop("verbose", False),
                **kwargs,
            )
        else:
            self.node_positions = graph_tool.draw.arf_layout(self.network.graph())


class RadialLayout(LayoutStrategy):
    def __init__(self, network: Network):
        super().__init__(network, "radial")
        self.node_angles = None
        self.node_angles_dict = None
        self.node_ratio = None

    def layout(self, *args, **kwargs):
        root_node = kwargs.get("root", 0)
        self.node_positions = graph_tool.draw.radial_tree_layout(
            self.network.graph(), root_node
        )

    def _post_layout(self):
        self.node_angles = np.degrees(
            np.arctan2(self.node_positions, self.node_positions)
        )
        self.node_angles_dict = dict(
            zip(self.node_angles_dict.keys(), self.node_angles)
        )
        self.node_ratios = np.sqrt(np.dot(self.node_positions, self.node_positions))

    def get_angle(self, idx):
        return self.node_angles_dict[int(idx)]

    def get_ratio(self, idx):
        return self.node_ratios[int(idx)]


class PrecomputedLayout(LayoutStrategy):
    def __init__(self, network: Network):
        super().__init__(network, "precomputed")

    def layout(self, *args, **kwargs):
        positions = np.array(kwargs.get("positions"))

        if positions.shape[0] != self.network.num_vertices():
            raise ValueError("dimensions do not match")

        self.node_positions = self.network.graph().new_vertex_property("vector<double>")
        for v, p in zip(self.network.vertices(), positions):
            self.node_positions[v] = p

        angles = kwargs.get("angles", None)
        ratios = kwargs.get("ratios", None)
        # print(angles, ratios)

        if angles is None and ratios is None:
            # do nothing
            return
        elif angles is not None and ratios is not None:
            self.node_ratios = ratios
            self.node_angles = angles
        else:
            raise ValueError("angles and ratios need to be provided simultaneously")

    def get_angle(self, idx):
        return getattr(self, "node_angles")[int(idx)]

    def get_ratio(self, idx):
        return getattr(self, "node_ratios")[int(idx)]


class GeographicalLayout(LayoutStrategy):
    def __init__(
        self, network: Network, geodataframe: gpd.GeoDataFrame, node_column: str = None
    ):
        super().__init__(network, name="geographical")
        self.node_column = node_column

        if len(self.network.node_map) > len(geodataframe):
            raise ValueError(f"GeoDataFrame has missing vertices")

        if self.node_column is None:
            self.geodf = geodataframe.loc[self.network.node_map.keys()].sort_index()
        else:
            self.geodf = geodataframe[
                geodataframe[node_column].isin(self.network.node_map.keys())
            ].sort_values(node_column)

        if len(self.network.node_map) != len(self.geodf):
            raise ValueError(
                f"Incompatible shapes: {len(self.network.node_map)} nodes and {len(self.geodf)} shapes. Do you have duplicate rows?"
            )

    def layout(self, *args, **kwargs):
        node_positions = positions_to_array(self.geodf.geometry.centroid)

        if len(node_positions) != len(self.network.node_map):
            raise ValueError(
                f"GeoDataFrame and Network have different lengths after filtering nodes. Maybe there are repeated values in the node column/index."
            )

        self.node_positions = node_positions
