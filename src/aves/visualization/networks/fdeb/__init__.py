import math
from collections import defaultdict
from itertools import combinations

import numpy as np
from cytoolz import sliding_window

from aves.features.geometry import euclidean_distance as point_distance
from aves.models.network import Network
from aves.models.network.edge import EPS, Edge


class FDB_Edge(object):
    def __init__(self, edge: Edge):
        self.base = edge

    def angle_compatibility(self, other):
        v1 = self.base.as_vector()
        v2 = other.base.as_vector()
        dot_product = np.dot(v1, v2)
        return max(0.0, dot_product / (self.base.length() * other.base.length()))

    def scale_compatibility(self, other):
        self_length = self.base.length()
        other_length = other.base.length()
        lavg = (self_length + other_length) / 2.0
        return 2.0 / (
            lavg / min(self_length, other_length)
            + max(self_length, other_length) / lavg
        )

    def position_compatibility(self, other):
        lavg = (self.base.length() + other.base.length()) / 2.0
        midP = self.base._mid_point
        midQ = other.base._mid_point

        return lavg / (lavg + point_distance(midP, midQ))

    def visibility(self, other, eps=EPS):
        I0 = self.base.project(other.base.source)
        I1 = self.base.project(other.base.target)

        divisor = point_distance(I0, I1)
        divisor = divisor if divisor > eps else eps

        midI = (I0 + I1) * 0.5
        return max(0, 1 - 2 * point_distance(self.base._mid_point, midI) / divisor)

    def visibility_compatibility(self, other):
        return min(self.visibility(other), other.visibility(self))

    def compatible_with(self, other, threshold=0.7):
        angles_score = self.angle_compatibility(other)
        scales_score = self.scale_compatibility(other)
        positi_score = self.position_compatibility(other)
        visibi_score = self.visibility_compatibility(other)

        score = angles_score * scales_score * positi_score * visibi_score

        return score >= threshold


class FDB:
    def __init__(
        self,
        network: Network,
        K=1,
        S=0.01,
        P=1,
        P_rate=2,
        C=6,
        I=70,
        I_rate=0.6666667,
        compatibility_threshold=0.7,
        eps=EPS,
    ):
        """
        @param K global bundling constant controlling edge stiffness
        @param S initial distance to move points
        @param P initial subdivision number
        @param P_rate subdivision rate increase
        @param C number of cycles to perform
        @param I initial number of iterations for cycle
        @param I_rate rate at which iteration numbers decreases i.e. 2/3
        """
        self.network = network
        self.edges = {}

        # Hyper-parameters
        self.K = K
        self.S_initial = S
        self.P_initial = P
        self.P_rate = P_rate
        self.C = C
        self.I_initial = I
        self.I_rate = I_rate
        self.compatibility_threshold = compatibility_threshold
        self.eps = EPS

        for base_edge in network.edge_data:
            edge = FDB_Edge(base_edge)

            if self.is_long_enough(edge):
                self.edges[base_edge.index] = edge

        # no compatibility by default
        self.compatible_edges = defaultdict(list)
        self.subdivision_points = defaultdict(list)

        # let's go
        self.bundle_edges()

        for base_edge in network.edge_data:
            if base_edge.index in self.subdivision_points:
                base_edge.points = np.array(self.subdivision_points[base_edge.index])

    def is_long_enough(self, edge):
        if np.allclose(edge.base.source, edge.base.target, atol=self.eps):
            return False

        raw_length = edge.base.length()

        if raw_length < (self.eps * self.P_initial * self.P_rate * self.C):
            return False
        else:
            return True

    def compute_compatibility_list(self):
        for e_idx, oe_idx in combinations(self.edges.keys(), 2):
            e1 = self.edges[e_idx]
            e2 = self.edges[oe_idx]

            if e1.compatible_with(e2, threshold=self.compatibility_threshold):
                self.compatible_edges[e_idx].append(oe_idx)
                self.compatible_edges[oe_idx].append(e_idx)

    def init_edge_subdivisions(self):
        for i in self.edges.keys():
            self.subdivision_points[i].append(self.edges[i].base.source)
            self.subdivision_points[i].append(self.edges[i].base.target)

    def compute_divided_edge_length(self, edge_idx):
        length = 0.0

        for p0, p1 in sliding_window(2, self.subdivision_points[edge_idx]):
            length += point_distance(p0, p1)

        return length

    def update_edge_divisions(self, P):
        for edge_idx in self.edges.keys():
            divided_edge_length = self.compute_divided_edge_length(edge_idx)
            segment_length = divided_edge_length / (P + 1)
            current_node = np.array(self.edges[edge_idx].base.source)
            new_subdivision_points = []
            number_subdiv_points = 0
            new_subdivision_points.append(np.array(current_node))
            # revisar que no se cambie si cambio el source
            number_subdiv_points += 1
            current_segment_length = segment_length
            i = 1
            finished = False
            while not finished:
                old_segment_length = point_distance(
                    self.subdivision_points[edge_idx][i], current_node
                )
                # direction is a vector of length = 1
                direction = (
                    self.subdivision_points[edge_idx][i] - current_node
                ) / old_segment_length

                if current_segment_length > old_segment_length:
                    current_segment_length -= old_segment_length
                    current_node = np.array(self.subdivision_points[edge_idx][i])
                    i += 1
                else:
                    current_node += current_segment_length * direction
                    new_subdivision_points.append(np.array(current_node))
                    number_subdiv_points += 1
                    current_segment_length = segment_length
                finished = number_subdiv_points == P + 1

            new_subdivision_points.append(np.array(self.edges[edge_idx].base.target))

            self.subdivision_points[edge_idx] = new_subdivision_points

    def apply_spring_force(self, edge_idx, i, kP):
        prev = self.subdivision_points[edge_idx][i - 1]
        next_ = self.subdivision_points[edge_idx][i + 1]
        crnt = self.subdivision_points[edge_idx][i]

        new_point = prev - crnt + next_ - crnt
        new_point[new_point < 0] = 0
        new_point *= kP
        return new_point

    def apply_electrostatic_force(self, edge_idx, i):
        sum_of_forces = np.array((0.0, 0.0))

        compatible_edges = self.compatible_edges[edge_idx]

        for oe in compatible_edges:
            force = (
                self.subdivision_points[oe][i] - self.subdivision_points[edge_idx][i]
            )

            if (math.fabs(force[0]) > self.eps) or (math.fabs(force[1]) > self.eps):
                divisor = point_distance(
                    self.subdivision_points[oe][i], self.subdivision_points[edge_idx][i]
                )
                diff = 1 / divisor

                sum_of_forces += force * diff

        return sum_of_forces

    def apply_resulting_forces_on_subdivision_points(self, edge_idx, K, P, S):
        # kP = K / | P | (number of segments), where | P | is the initial length of edge P.
        kP = K / (self.edges[edge_idx].base.length() * (P + 1))

        # (length * (num of sub division pts - 1))
        resulting_forces_for_subdivision_points = []
        resulting_forces_for_subdivision_points.append(np.array((0.0, 0.0)))

        for i in range(1, P + 1):  # exclude initial end points of the edge 0 and P+1
            spring_force = self.apply_spring_force(edge_idx, i, kP)
            electrostatic_force = self.apply_electrostatic_force(edge_idx, i)

            resulting_force = S * (spring_force + electrostatic_force)
            resulting_forces_for_subdivision_points.append(resulting_force)

        resulting_forces_for_subdivision_points.append(np.array((0.0, 0.0)))
        return resulting_forces_for_subdivision_points

    def bundle_edges(self):
        S = self.S_initial
        I = self.I_initial
        P = self.P_initial

        self.init_edge_subdivisions()
        self.compute_compatibility_list()
        self.update_edge_divisions(P)

        for _cycle in range(self.C):
            for _iteration in range(math.ceil(I)):
                forces = {}
                for edge_idx in self.edges.keys():
                    forces[
                        edge_idx
                    ] = self.apply_resulting_forces_on_subdivision_points(
                        edge_idx, self.K, P, S
                    )

                for edge_idx in self.edges.keys():
                    for i in range(P + 1):  # We want from 0 to P
                        self.subdivision_points[edge_idx][i] = (
                            self.subdivision_points[edge_idx][i] + forces[edge_idx][i]
                        )

            # prepare for next cycle
            S = S / 2
            P = P * self.P_rate
            I = I * self.I_rate

            self.update_edge_divisions(P)
