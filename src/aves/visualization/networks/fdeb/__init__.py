import math
from collections import defaultdict
from itertools import combinations

import numpy as np
from cytoolz import sliding_window

from aves.features.geometry import euclidean_distance as point_distance
from aves.models.network import Network
from aves.models.network.edge import EPS, Edge


class FDB_Edge(object):
    """
    Una arista para el cálculo de Edge Bundling con Fuerza-Direccional (FDB).

    Attributes
    ----------
    base : Edge
        El objeto de arista base a partir del cual se construye la arista FDB.
    """
    def __init__(self, edge: Edge):
        """
        Inicializa el objeto.

        Parameters
        ------------
        edge : Edge
            La arista que se va a representar.
        """
        self.base = edge

    def angle_compatibility(self, other):
        """
        Calcula la compatibilidad de ángulo con otra arista. Mientras más cercano a 1 es el valor retornado, más simialres son las aristas en
        cuanto a dirección.

        Parameters
        ------------
        other : FDB_Edge
            La arista con la cual se calcula la compatibilidad.

        Returns
        -------
        float
            El puntaje de compatibilidad de ángulo.
        """
        v1 = self.base.as_vector()
        v2 = other.base.as_vector()
        dot_product = np.dot(v1, v2)
        return max(0.0, dot_product / (self.base.length() * other.base.length()))

    def scale_compatibility(self, other):
        """
        Calcula la compatibilidad de escala con otra arista. Este valor es un indicador de qué tan similares son los
        largos de las aristas, un valor más alto indica que las aristas son compatibles para ser agrupadas en cuanto a
        escala.

        Parameters
        ------------
        other : FDB_Edge
            La arista con la cual se calcula la compatibilidad.

        Returns
        -------
        float
            El puntaje de compatibilidad de escala.
        """
        self_length = self.base.length()
        other_length = other.base.length()
        lavg = (self_length + other_length) / 2.0
        return 2.0 / (
            lavg / min(self_length, other_length)
            + max(self_length, other_length) / lavg
        )

    def position_compatibility(self, other):
        """
        Calcula la compatibilidad de posición entre esta arista y otra. Un valor más alto de compatibilidad
        de posición indica que las aristas tienen posiciones cercanas entre sí en el plano y, por lo tanto, son
        beunas candidatas para ser agrupadas en la visualización.

        Parameters
        ------------
        other : FDB_Edge
            La arista con la cual se comparará.

        Returns
        -------
        float
            El puntaje de compatibilidad de posición.
        """
        lavg = (self.base.length() + other.base.length()) / 2.0
        midP = self.base._mid_point
        midQ = other.base._mid_point

        return lavg / (lavg + point_distance(midP, midQ))

    def visibility(self, other, eps=EPS):
        """
        Calcula la visibilidad entra la arista y la arista especificada. Esto es un indicador cuánto de una arista
        es visible o se puede ver en relación con otra arista cercana en el grafo.

        Parameters
        ------------
        other : FDB_Edge
            La arista con la cual se hará la comparación.
        eps : float, opcional
            El valor epsilon para evitar la división por cero, por defecto EPS.

        Returns
        -------
        float
            El puntaje de visibilidad.
        """
        I0 = self.base.project(other.base.source)
        I1 = self.base.project(other.base.target)

        divisor = point_distance(I0, I1)
        divisor = divisor if divisor > eps else eps

        midI = (I0 + I1) * 0.5
        return max(0, 1 - 2 * point_distance(self.base._mid_point, midI) / divisor)

    def visibility_compatibility(self, other):
        """
        Retorna la compatibilidad de visibilidad entre esta arista y otra.

        Parameters
        ------------
        other : FDB_Edge
            La arista con cual se hará la comparación.

        Returns
        -------
        float
            El puntaje de compatibilidad de visibilidad.
        """
        return min(self.visibility(other), other.visibility(self))

    def compatible_with(self, other, threshold=0.7):
        """
        Determina si la arista es compatible con otra, a partir de los distintos puntajes de compatibilidad
        obtenidos por el par de aristas. Si el producto entre todos los puntajes da un valor mayor al umbral
        de compatibilidad, entonces las aristas son compatibles y por ende buenas candidatas para
        ser agrupadas en la visualizaciónd el grafo,

        Parameters
        ------------
        other : FDB_Edge
            La arista con la que se busca determinar la compatibilidad.
        threshold : float, default=0.75, optional
            El umbral de compatibilidad.

        Returns
        -------
        bool
            Verdadero si las aristas son compatibles, Falso en caso contrario.
        """
        angles_score = self.angle_compatibility(other)
        scales_score = self.scale_compatibility(other)
        positi_score = self.position_compatibility(other)
        visibi_score = self.visibility_compatibility(other)

        score = angles_score * scales_score * positi_score * visibi_score

        return score >= threshold


class FDB:
    """
    Construye la visualización de la red con agrupamiento de aristas según la heurística "force directed".
    Esta agrupa las aristas que se parecen según distintos criterios y evita que las aristas de distintos grupos se toquen.
    Puedes encontrar más información sobre este método en la publicación `Force-Directed Edge Bundling for Graph Visualization <https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1467-8659.2009.01450.x>`__.
   
    Para ejemplos de uso de esa clase, revisa el código del `taller de mapas en Python </https://github.com/zorzalerrante/aves/blob/master/notebooks/talleres/zorzal-tv-01-mapas.py>`_.

    Attributes
    -----------
    
    network : Network
        La red a la que se aplicará el agrupamiento de aristas.
    edges : dict
        Diccionario que mapea índices de aristas a objetos FDB_Edge.
    
      """
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
        Inicializa el algoritmo de Edge Bundling basado en la fuerza dirigida.

        Parameters
        -----------
            network : Network
                La red en la que se aplicará el Edge Bundling.
            K : int, default=1
                Constante global de agrupamiento que controla la rigidez de las aristas.
            S : float, default=0.01
                Distancia inicial para mover los puntos en el algoritmo.
            P : int, default=1
                Número inicial de subdivisiones.
            P_rate : float, default=2
                Tasa de aumento en la subdivisión.
            C : int, default=6
                Número de ciclos a realizar.
            I : int, default=70
                Número inicial de iteraciones por ciclo.
            I_rate : float, default=0.6666667
                Tasa a la cual disminuyen los números de iteración (por ejemplo, 2/3).
            compatibility_threshold : float, default=0.7
                Umbral de compatibilidad para considerar aristas como compatibles.
            eps : float, default=EPS
                Pequeño valor de epsilon para cálculos numéricos.

        Notes
        -------
            - El algoritmo trabaja con aristas lo suficientemente largas (longitud mayor a EPS).
            - El método bundle_edges() se llama para realizar el proceso de agrupamiento de aristas.
            - Los puntos de subdivisión se asignan a las aristas base según corresponda.
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
        """
        Determina si una arista es suficientemente larga para ser procesada.

        Parameters
        ----------
        edge : FDB_Edge
            La arista a revisar.

        Returns
        -------
        bool
            True si la arista es suficientemente larga, False si no.
        """
        if np.allclose(edge.base.source, edge.base.target, atol=self.eps):
            return False

        raw_length = edge.base.length()

        if raw_length < (self.eps * self.P_initial * self.P_rate * self.C):
            return False
        else:
            return True

    def compute_compatibility_list(self):
        """
        Calcula la lista de aristas compatibles para ser agrupadas.

        Este método llena el atributo :attr:`~aves.visualization.networks.fdeb.FDB.compatible_edges` con pares de índices de aristas
        que son compatibles.
        """
        for e_idx, oe_idx in combinations(self.edges.keys(), 2):
            e1 = self.edges[e_idx]
            e2 = self.edges[oe_idx]

            if e1.compatible_with(e2, threshold=self.compatibility_threshold):
                self.compatible_edges[e_idx].append(oe_idx)
                self.compatible_edges[oe_idx].append(e_idx)

    def init_edge_subdivisions(self):
        """
        Inicializa los puntos de subdivisión de las aristas.

        Este método modifica el atributo :attr:`~aves.visualization.networks.fdeb.FDB.subdivision_points`.
        """
        for i in self.edges.keys():
            self.subdivision_points[i].append(self.edges[i].base.source)
            self.subdivision_points[i].append(self.edges[i].base.target)

    def compute_divided_edge_length(self, edge_idx):
        """
        Calcula el largo total de una arista dividida.

        Parameters
        ----------
        edge_idx : int
            Índice de la arista en :attr:`~aves.visualization.networks.fdeb.FDB.subdivision_points`.

        Returns
        -------
        float
            Largo total de la arista dividida.
        """
        length = 0.0

        for p0, p1 in sliding_window(2, self.subdivision_points[edge_idx]):
            length += point_distance(p0, p1)

        return length

    def update_edge_divisions(self, P):
        """
        Actualiza la subdivisión de las aristas según el nivel de subdivisión especificado.

        Parameters
        ----------
        P : int
            El nivel de subdivisión.

        Returns
        -------
        None
        """
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
        """
        Aplica fuerza de resorte a un punto de subdivisión.

        Este método calcula la nueva posición de un punto de subdivisión en función de la fuerza de
        resorte aplicada por sus puntos vecinos. La posición resultante se escala mediante el factor kP.

        Parameters
        ------------
        edge_idx : int
            El índice de la arista.
        i : int
            El índice del punto de subdivisión que se actualizará.
        kP : float
            El factor de escala para la fuerza de resorte.

        Returns
        ---------
        new_point : numpy.ndarray
            La nueva posición del punto de subdivisión.
        """
        prev = self.subdivision_points[edge_idx][i - 1]
        next_ = self.subdivision_points[edge_idx][i + 1]
        crnt = self.subdivision_points[edge_idx][i]

        new_point = prev - crnt + next_ - crnt
        new_point[new_point < 0] = 0
        new_point *= kP
        return new_point

    def apply_electrostatic_force(self, edge_idx, i):
        """
        Aplica "fuerza electrostática" a un punto de subdivisión. Fuerza electrostática se refiere a una fuerza
        simulada que modela la interacción repulsiva entre los puntos de subdivisión de diferentes aristas.

        Este método calcula la suma de las fuerzas electrostáticas ejercidas por los puntos de subdivisión
        compatibles de otras aristas.

        Parameters
        ----------
        edge_idx : int
            El índice de la arista.
        i : int
            El índice del punto de subdivisión para el cual se calculará la fuerza.

        Returns
        -------
        sum_of_forces : numpy.ndarray
            La suma de las fuerzas electrostáticas ejercidas por los puntos de subdivisión compatibles de otras aristas.
        """
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
        """
        Aplica las fuerzas resultantes a los puntos de subdivisión de una arista.

        Esta función calcula y aplica las fuerzas resultantes (fuerzas de resorte y fuerzas electrostáticas)
        a los puntos de subdivisión de una arista específica.

        Parameters
        ------------
        edge_idx : int
            Índice de la arista en la que se aplicarán las fuerzas.
        K : float
            Constante global de agrupamiento que controla la rigidez de las aristas.
        P : int
            Número de subdivisiones iniciales de la arista.
        S : float
            Distancia inicial para mover los puntos de subdivisión.

        Returns
        --------
        resulting_forces_for_subdivision_points : list of numpy arrays
            Lista de fuerzas resultantes aplicadas a cada punto de subdivisión de la arista, incluyendo los puntos de los extremos.
        """
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
        """
        Agrupa las aristas utilizando el algoritmo de agrupamiento de aristas dirigido por fuerzas.

        Notes
        -------
        El método utiliza los hiperparámetros especificados en la inicialización del objeto FDB.

        """
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
