from abc import ABC, abstractmethod

import geopandas as gpd
import graph_tool
import graph_tool.draw
import graph_tool.topology
import numpy as np

from aves.features.geo import positions_to_array

from .base import Network


class LayoutStrategy(ABC):
    """
    Clase abstracta que representa un algoritmo de organización (layout algorithm) de nodos. Estos algoritmos posicionan los nodos de
    la red en un plano, siguiendo distintas heurísticas, a modo de visualizar la red.

    Attributes
    ----------
    network : Network
        La red sobre la cual se aplica el algoritmo.
    name : str
        El nombre del método de posicionamiento.
    node_positions : `~graph_tool.VertexPropertyMap`
        Las coordenadas de los vértices en el diseño, o None si el diseño aún no ha sido calculado.
    node_positions_dict : dict
        Un diccionario que mapea un vértice (a través de su id) a su posición en el diseño, o None si el diseño aún no ha sido calculado.
    node_positions_vector : np.array
        Un vector numpy que contiene las posiciones de los vértices en el diseño, o None si el diseño aún no ha sido calculado.

    """

    def __init__(self, network: Network, name: str):
        """
        Parameters
        ----------
        network : Network
            La red sobre la cual se aplicará el algoritmo de organización de nodos.
        name : str
            El nombre del método de posicionamiento de nodos.
        """
        self.network = network
        self.name = name
        self.node_positions = None
        self.node_positions_dict: dict = None
        self.node_positions_vector: np.array = None

    @abstractmethod
    def layout(self):
        """
        Método abstracto implementado por las clases que heredan de LayoutStrategy.
        Este método realiza el cálculo de las posiciones de los nodos según la heurística de cada subclase.
        """
        pass

    def _post_layout(self):
        """
        Método que puede ser sobrescrito por las clases que heredan de LayoutStrategy para realizar
        acciones adicionales después de calcular el diseño.
        """
        pass

    def layout_nodes(self, *args, **kwargs):
        """
        Calcula las posiciones de los nodos de la red llamando al método `layout` y setea los
        atributos `node_positions_dict` y `node_positions_vector` con el resultado de aplicar el algoritmo organizacional.

        Parameters
        ------------
        *args: positional arguments
            argumentos requeridos por la función `layout`.
        **kwargs: keyword arguments
            Argumentos que se pasarán a la función `layout`.
        """
        self.layout(*args, **kwargs)

        self.node_positions_vector = np.array(list(self.node_positions))
        self.node_positions_dict = dict(
            zip(
                list(map(int, self.network.vertices)),
                list(self.node_positions_vector),
            )
        )

        self._post_layout()

        return self.node_positions

    def get_position(self, idx):
        """
        Obtiene la posición (coordenadas) de un nodo dado su identificador.

        Parameters
        ----------
        idx : int
            El id del nodo

        Returns
        -------
        tuple
            La posición (x, y) del nodo en el diseño.
        """
        idx = int(idx)
        return self.node_positions_dict[idx]

    def get_angle(self, idx):
        """
        Obtiene la posición angular de un nodo en la red, en el caso de tratarse de una organización de tipo árbol radial ("Radial Tree").
        El ángulo retornado se encuentra en grados.

        Parameters
        ----------
        idx : int
            El identificador del nodo.

        Raises
        ------
        NotImplementedError
            Si el tipo de layout/organización de nodos del grafo no contempla el ángulo de un nodo.

        Returns
        -------
        float
            El ángulo del nodo especificado.
        """
        raise NotImplementedError("this class doesn't work with angles")

    def get_ratio(self, idx):
        """
        Obtiene el ratio de un nodo en la red. El ratio de un nodo se refiere a la distancia relativa de un nodo con respecto
        al nodo raíz del árbol en una organización de nodos de tipo árbol radial ("Radial Tree").
        En este caso, el ratio corresponde a la distancia euclidiana entre un nodo y el nodo raíz (centro).

        Parameters
        ----------
        idx : int
            El identificador del nodo.

        Raises
        ------
        NotImplementedError
            Si el tipo de layout/organización de nodos del grafo no contempla el ratio de un nodo.

        Returns
        -------
        float
            El ratio del nodo especificado.
        """
        raise NotImplementedError("this class doesn't work with ratios")

    def positions(self):
        """
        Devuelve las posiciones de todos los nodos de la red.

        Returns
        -------
        np.array
            Arreglo que contiene las coordenadas de cada nodo.
        """
        return self.node_positions_vector


class ForceDirectedLayout(LayoutStrategy):
    """
    Estrategia de posicionamiento de nodos que utiliza el algoritmo de layout "Force Directed" ("sfdp").
    Este algoritmo posiciona los nodos de un grafo de tal manera que los nodos con conexiones más fuertes
    se coloquen más cerca entre sí, mientras que los nodos con conexiones más débiles o sin conexiones se mantengan más separados.

    Parameters
    ----------
    network : Network
        La red sobre la cual se aplicará el algoritmo.

    """

    def __init__(self, network: Network):
        super().__init__(network, "force-directed")

    def layout(self, *args, **kwargs):
        """
        Calcula las posiciones de los nodos de la red utilizando la heurística "Force Directed".
        Al finalizar el algoritmo, las posiciones de los nodos se almacenan en el atributo `node_positions`, reemplazando el valor previo.

        Parameters
        ------------
        **kwargs: keyword arguments
            Parámetros adicionales que permiten configurar la ejecución del algoritmo. Una lista completa de los argumentos
            disponibles se encuentra en la documentación de `graph-tool <https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.draw.sfdp_layout.html#graph_tool.draw.sfdp_layout>`__.
        """
        method = kwargs.pop("algorithm", "sfdp")

        if not method in ("sfdp", "arf"):
            raise ValueError(f"unsupported method: {method}")

        if method == "sfdp":
            self.node_positions = graph_tool.draw.sfdp_layout(
                self.network.graph,
                eweight=self.network._edge_weight,
                verbose=kwargs.pop("verbose", False),
                **kwargs,
            )
        else:
            self.node_positions = graph_tool.draw.arf_layout(self.network.graph)


class RadialLayout(LayoutStrategy):
    """
    Estrategia de posicionamiento de nodos que utiliza el algoritmo de layout "Árbol Radial" ("radial").
    Este algoritmo posiciona los nodos de un grafo en círculos concéntricos alrededor del nodo raíz,
    y coloca las conexiones de la raíz en el primer círculo alrededor de esta. Luego, para cada nodo en
    este círculo, coloca sus conexiones en círculos exteriores y así sucesivamente, siguiendo la jerarquía del árbol.
    Este algoritmo se basa en el árbol recubridor mínimo (minimum spanning tree) del grafo.

    Parameters
    ----------
    network : Network
        La red sobre la cual se aplicará el algoritmo.

    """
    def __init__(self, network: Network):
        super().__init__(network, "radial")
        self.node_angles = None
        self.node_angles_dict = None
        self.node_ratio = None

    def layout(self, *args, **kwargs):
        """
        Posiciona los nodos usadno el algoritmo de árbol radial. Este método almacena las posiciones resultantes en
        el atributo `node_positions`.

        Parameters
        ----------
        root : int, optional
            El id del nodo raíz alrededor del cual se organizarán los otros nodos. Por defecto es 0.
            
        **kwargs: keyword arguments
            Parámetros adicionales que permiten configurar la ejecución del algoritmo. Una lista completa de los argumentos
            disponibles se encuentra en la documentación de `graph-tool <https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.draw.sfdp_layout.html#graph_tool.draw.radial_tree_layout>`__.

        Returns
        -------
        None
        """
        root_node = kwargs.get("root", 0)
        self.node_positions = graph_tool.draw.radial_tree_layout(
            self.network.graph, root_node
        )

    def _post_layout(self):
        """
        Ejecuta acciones adicionales luego de calcular el layout de árbol radial.

        Este método calcula el ángulo de cada nodo, que representa su posición angular en el círculo. Crea un diccionario que mapea
        los ids de los nodos a su ángulo en el diseño, y lo almacena en el atributo `node_angles_dict`.

        También calcula el `ratio` de un nodo, que representa su distancia al centro del diseño. Mientras mayor es el `ratio` de un nodo,
         mayor es la distancia hasta la raíz. Almacena esta información en el atributo `node_ratios`.

        Returns
        -------
        None
        """
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
    """
    Organización de nodos a partir de posiciones previamente calculadas.
    """

    def __init__(self, network: Network):
        """
        Inicializa una estrategia de diseño que utiliza posiciones de nodos precalculadas.

        Parameters
        ----------
        network : Network
            La red sobre la cual se aplicará la estrategia de diseño.
        """
        super().__init__(network, "precomputed")

    def layout(self, *args, **kwargs):
        """
        Establece las posiciones de los nodos basándose en posiciones precalculadas.

        Este método establece las posiciones de los nodos en el diseño utilizando las
        coordenadas precalculadas proporcionadas en el parámetro `positions`.

        Parameters
        ----------
        *args : positional arguments
            Argumentos posicionales adicionales (sin uso).
        **kwargs : keyword arguments
            Argumentos de palabras clave adicionales.
                - positions : numpy.ndarray
                    Un array de posiciones de nodos con forma `(num_nodos, 2)`.
                    Cada fila contiene las coordenadas (x, y) de un nodo.
                - angles : numpy.ndarray, opcional
                    Un array de ángulos de nodos, donde cada ángulo representa la posición angular de un
                    nodo en el diseño. Si no se proporciona, se establecerá en `None`.
                - ratios : numpy.ndarray, opcional
                    Un array de relaciones de nodos, donde cada relación representa la distancia de un nodo
                    desde la raíz del árbol. Si no se proporciona, se establecerá en `None`.

        Raises
        ------
        ValueError
            Si las dimensiones de las posiciones precalculadas no coinciden con el número de nodos de la red.
        ValueError
            Si se proporcionan solo ángulos o ratios, pero no ambos simultáneamente.

        Returns
        -------
        None
        """
        positions = np.array(kwargs.get("positions"))

        if positions.shape[0] != self.network.num_vertices:
            raise ValueError("dimensions do not match")

        self.node_positions = self.network.graph.new_vertex_property("vector<double>")
        for v, p in zip(self.network.vertices, positions):
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
    """
    Estrategia de organización de nodos que se basa en coordenadas geográficas para posicionar los nodos.
    """
    
    def __init__(
        self, network: Network, geodataframe: gpd.GeoDataFrame, node_column: str = None
    ):
        """
        Inicializa una estrategia de organización de nodos que utiliza posiciones geográficas para posicionar los nodos.

        Parameters
        ----------
        network : Network
            La red a la cual se aplicará la estrategia de diseño.
        geodataframe : gpd.GeoDataFrame
            Un GeoDataFrame que contiene las geometrías geográficas de los nodos.
        node_column : str, opcional
            El nombre de la columna que contiene los identificadores de los nodos en el GeoDataFrame.
            Si no se proporciona, se asume que los identificadores de los nodos se encuentran en el índice del GeoDataFrame.

        Raises
        ------
        ValueError
            Si el GeoDataFrame tiene nodos faltantes en comparación con la red.
        ValueError
            Si el GeoDataFrame y la red tienen longitudes diferentes después de filtrar los nodos.
            Esto puede suceder si hay filas duplicadas en la columna de identificadores de nodos.
        """
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
        """
        Establece las posiciones de los nodos en el diseño utilizando las coordenadas geográficas
        del GeoDataFrame proporcionado. Las posicions quedan almacenadas en el atributo `node_positions` de la red.

        Parameters
        ------------
        *args : positional arguments
            Argumentos posicionales adicionales (sin uso).
        **kwargs : keyword arguments
            Argumentos de palabras clave adicionales (sin uso).

        Raises
        ------
        ValueError
            Si las dimensiones de las posiciones geográficas no coinciden con el número de nodos de la red.

        Returns
        --------
        None
        """
        node_positions = positions_to_array(self.geodf.geometry.centroid)

        if len(node_positions) != len(self.network.node_map):
            raise ValueError(
                f"GeoDataFrame and Network have different lengths after filtering nodes. Maybe there are repeated values in the node column/index."
            )

        self.node_positions = node_positions
