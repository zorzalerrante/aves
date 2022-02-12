import numpy as np

EPS = 1e-6


class Edge(object):
    def __init__(self, source, target, source_idx, target_idx, weight=None, index=-1):
        self.source = source
        self.target = target

        self._vector = self.target - self.source

        if np.allclose(self.source, self.target, atol=EPS):
            self._length = EPS
        else:
            self._length = np.sqrt(np.dot(self._vector, self._vector))

        self._unit_vector = self._vector / self._length
        self._mid_point = (self.source + self.target) * 0.5

        if weight is None:
            self.weight = 1
        else:
            self.weight = weight

        self.index = index
        # this can be filled by y external algorithms
        self.points = [self.source, self.target]
        self.index_pair = (source_idx, target_idx)

    def as_vector(self):
        return self._vector

    def length(self):
        return self._length

    def project(self, point):
        L = self._length
        p_vec = point - self.source
        return self.source + np.dot(p_vec, self._unit_vector) * self._unit_vector
