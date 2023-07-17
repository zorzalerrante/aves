import numpy as np

EPS = 1e-6


class Edge(object):
    """
    Una arista que conecta dos vértices en un espacio bidimensional.

    Attributes
    --------------
    source : ArrayLike
        Posicion de vértice de origen de la arista.
    target : ArrayLike
        Posicion de vértice de destino de la arista.
    index : int
        Índice de la arista.
    points : list[]
        Lista de vértices que conforman la arista.
    index_pair : tuple[int, int]
        Tupla que contiene los índices del origen y el destino.
    _vector : numpy.ndarray
        Vector que representa el desplazamiento desde el origen al destino.
    _length : float
        Longitud de la arista.
    _unit_vector : numpy.ndarray
        Vector unitario que representa la dirección de la arista.
    _mid_point : numpy.ndarray
        Punto medio de la arista.

    """
    def __init__(self, source, target, source_idx, target_idx, index=-1):
        """
        
        """
        self.source = source
        self.target = target

        self._vector = self.target - self.source

        if np.allclose(self.source, self.target, atol=EPS):
            self._length = EPS
        else:
            self._length = np.sqrt(np.dot(self._vector, self._vector))

        self._unit_vector = self._vector / self._length
        self._mid_point = (self.source + self.target) * 0.5

        self.index = index
        # this can be filled by y external algorithms
        self.points = [self.source, self.target]
        self.index_pair = (source_idx, target_idx)

    def as_vector(self):
        """
        Devuelve el vector unitario que representa la arista. El vector se calcula como el desplazamiento entre el origen y el destino.

        Returns
        ---------
        numpy.ndarray
            Vector que representa la arista.
        """
        return self._vector

    def length(self):
        """
        Devuelve la longitud de la arista. Se define la longitud como la raíz cuadrada del producto
        punto del vector de desplazamiento entre los vértices.
        Si los vértices son puntos cercanos según una tolerancia definida como 1e-6, se establece
        una longitud mínima equivalente a esta para evitar divisiones por cero.

        Returns
        ----------
        float
            Longitud de la arista.
        """

        return self._length

    def project(self, point):
        """
        Proyecta un punto en la arista.

        Parameters
        ------------
        point : numpy.ndarray
            Punto a proyectar.

        Returns
        ---------
        numpy.ndarray
            Punto proyectado.
        """
        L = self._length
        p_vec = point - self.source
        return self.source + np.dot(p_vec, self._unit_vector) * self._unit_vector
