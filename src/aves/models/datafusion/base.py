import random
from itertools import chain

import numpy as np
import pandas as pd
from cytoolz import itemmap, sliding_window, valmap
from skfusion import fusion


class DataFusionModel(object):
    def __init__(
        self, nodes, relations, init_type="random", random_state=666, n_jobs=1
    ):
        self.nodes = nodes
        self.relation_definitions = relations

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.init_type = init_type

    def reconstruct(self, src, dst, idx=0, return_dataframe=True):
        relation = list(
            self.fuser.fusion_graph.get_relations(self.types[src], self.types[dst])
        )[idx]
        values = self.fuser.complete(relation)

        if return_dataframe:
            components = self.relation_definitions[(src, dst)][idx]
            return pd.DataFrame(
                values, index=components.index.values, columns=components.columns.values
            )

        return values

    def factor(self, type_name, return_dataframe=True):
        factor = self.fuser.factor(self.types[type_name])
        if not return_dataframe:
            return factor

        profile = pd.DataFrame(
            factor,
            index=self.indices[type_name],
            columns=[f"C{i:02}" for i in range(factor.shape[1])],
        )
        return profile

    def _construct_relationship(self, path, updated_factors):
        start_node = path[0]
        end_node = path[-1]

        computed_matrix = (
            self.fuser.factor(start_node)
            if not start_node.name in updated_factors
            else updated_factors[start_node.name]
        )
        print(
            type(start_node),
            start_node,
            start_node.name in updated_factors,
            computed_matrix.shape,
        )

        for src, dst in sliding_window(2, path):
            relation = list(self.fuser.fusion_graph.get_relations(src, dst))[0]
            print(relation)
            computed_matrix = np.dot(computed_matrix, self.fuser.backbone(relation))

        end_factor = (
            self.fuser.factor(end_node)
            if not end_node.name in updated_factors
            else updated_factors[end_node.name]
        )
        computed_matrix = np.dot(computed_matrix, end_factor.T)

        return computed_matrix

    def relation_profiles(self, src, dst, updated_factors=None, index=None):
        if updated_factors is None:
            updated_factors = {}
        if index is None:
            index = self.indices[src]

        paths = list(self.fuser.chain(self.types[src], self.types[dst]))

        relations = []
        for path in paths:
            rel = self._construct_relationship(path, updated_factors)
            profile = pd.DataFrame(rel, index=index, columns=self.indices[dst])
            relations.append(profile)

        return list(zip(paths, relations))

    def fit(self, method='factorization'):
        self.types = dict(
            zip(
                self.nodes.keys(),
                map(lambda x: fusion.ObjectType(*x), self.nodes.items()),
            )
        )
        print(self.types)

        self.relations = map(
            lambda x: map(
                lambda r: fusion.Relation(
                    r.values, self.types[x[0][0]], self.types[x[0][1]]
                ),
                x[1],
            ),
            self.relation_definitions.items(),
        )
        self.relations = list(chain(*self.relations))
        print(self.relations)

        self.indices = {}
        for (src, dst), dfs in self.relation_definitions.items():
            if not src in self.indices:
                self.indices[src] = list(dfs[0].index)
            if not dst in self.indices:
                self.indices[dst] = list(dfs[0].columns)

        random.seed(self.random_state)
        np.random.seed(self.random_state)

        self.fusion_graph = fusion.FusionGraph(self.relations)

        if method == 'factorization':
            fuser = fusion.Dfmf
        elif method == 'completion':
            fuser = fusion.Dfmc
        else:
            raise ValueError('method must be factorization or completion')

        self.fuser = fuser(
            init_type=self.init_type, random_state=self.random_state, n_jobs=self.n_jobs
        )

        self.fuser.fuse(self.fusion_graph)
