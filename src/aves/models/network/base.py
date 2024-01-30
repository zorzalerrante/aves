from typing import Dict, Optional

import graph_tool
import graph_tool.centrality
import graph_tool.inference
import graph_tool.topology
import graph_tool.search
import numpy as np
import pandas as pd
from cytoolz import itemmap, valfilter, valmap
import joblib
from collections import defaultdict
import os.path
from pathlib import Path

from .edge import Edge


class Network(object):
    """
    Una red de vértices conectados por aristas.

    Attributes
    ----------------
        network:  `graph_tool.Graph <https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.Graph.html>`__
            El grafo que almacena la estructura de la red.
        edge_data: List[Edge]
            Lista con la información de las aristas de la red.
        node_map: dict
            Un diccionario que mapea identificadores de nodos a vértices en la red.
        node_layout: LayoutStrategy
            La estrategia de diseño usada para posicionar los nodos al visualizar la red.
        id_to_label: dict
            Diccionario que mapea el identificador de un nodo a su etiqueta.
        community_tree: graph_tool.Graph
            La estructura jerárquica de las comunidades detectadas en la red.
        community_root: int
            El nodo raíz en el árbol de comunidades.
    """
    def __init__(self, graph: graph_tool.Graph):
        """
        Inicializa una nueva instancia de la clase Network.

        Parameters
        ----------
        graph : graph_tool.Graph
            Objeto Graph de graph-tool que representa la red.

        Returns
        -------
        Network
            Una nueva instancia de la clase Network.
        """
        self.network: graph_tool.Graph = graph
        self.edge_data = None

        self.node_map: dict = None
        self.node_layout: LayoutStrategy = None

        self.id_to_label: dict = None
        self.community_tree = None
        self.community_root = None

    @classmethod
    def load(cls, filename: str):
        """
        Carga una red desde un archivo.

        Parameters
        ----------
        filename : str
            Ruta del archivo que contiene la red. Ela rchivo debe estar en formato “gt”, “graphml”, “xml”, “dot” o “gml”.
            Graph_tool recomienda usar "gt" o "graphml" para garantizar la conservación de las propiedades internas del grafo cargado.

        Returns
        ---------
        Network
            Instancia de la clase Network creada a partir del archivo.
        """
        if isinstance(filename, Path):
            filename = str(filename)

        network = graph_tool.Graph()
        network.load(filename)

        result = Network(network)
        result.node_map = dict(
            zip(network.vertex_properties["elem_id"], network.vertices())
        )
        result.id_to_label = itemmap(reversed, result.node_map)

        tree_filename = filename + ".community_tree.gt"
        if os.path.exists(tree_filename):
            result.community_tree = graph_tool.Graph()
            result.community_tree.load(tree_filename)
            result.community_root = result.community_tree.graph_properties["root_node"]
            result.communities_per_level = result._build_node_memberships()

        return result

    def save(self, filename: str):
        """
        Guarda la red en un archivo.

        Parameters
        ----------
        filename : str
            Ruta del archivo donde se guardará la red. El archivo debe ser formato "gt", es decir, terminar en ".gt".
        """
        if isinstance(filename, Path):
            filename = str(filename)

        self.network.save(filename, fmt="gt")

        if self.community_tree is not None:
            self.community_tree.save(filename + ".community_tree.gt", fmt="gt")

    @classmethod
    def from_edgelist(
        cls,
        df: pd.DataFrame,
        source="source",
        target="target",
        directed=True,
        weight=None,
        allow_negative_weights=False,
        properties=None,
    ):
        """
        Crea una red a partir de un listado de aristas.

        Parameters
        ------------
            df: pandas.DataFrame
                El DataFrame que contiene la lista de aristas.
            source: str, default="source"
                El nombre de la columna de `df` que contiene los nodos de origen.
            target: str, default="target"
                El nombre de la columna de `df` que contiene los nodos de destino.
            directed: bool, default=True
                Indica si el grafo es dirigido o no.
            weight: str, default=None
                El nombre de la columna de `df` que contiene los pesos de las aristas de existir.
        Returns
        ----------
            Network: La red creada
        """
        source_attr = f"{source}__mapped__"
        target_attr = f"{target}__mapped__"

        node_values = set(df[source].unique())
        node_values = sorted(node_values | set(df[target].unique()))
        node_map = dict(zip(node_values, range(len(node_values))))

        df_mapped = df.assign(
            **{
                source_attr: df[source].map(node_map),
                target_attr: df[target].map(node_map),
            }
        )

        network = cls._parse_edgelist(
            df_mapped,
            source_attr,
            target_attr,
            weight_column=weight,
            directed=directed,
            allow_negative_weights=allow_negative_weights,
        )

        network.vertex_properties["elem_id"] = network.new_vertex_property(
            "object", vals=node_values
        )

        result = cls(network)
        result.node_map = node_map
        result.id_to_label = itemmap(reversed, node_map)

        if properties is not None:
            for prop in properties:
                if df[prop].dtype in (np.int8, np.int16, np.int32, np.int64):
                    dtype = "int"
                elif df[prop].dtype in (np.float16, np.float32, np.float64):
                    dtype = "float"
                else:
                    raise ValueError(
                        f"property {prop} has an unsupported dtype {prop.dtype}"
                    )
                result.add_edge_property(
                    df.set_index([source, target])[prop], dtype=dtype
                )

        return result

    @classmethod
    def _parse_edgelist(
        cls,
        df,
        source_column,
        target_column,
        weight_column=None,
        directed=True,
        allow_negative_weights=False,
    ) -> graph_tool.Graph:
        """Crea un grafo a partir de un listado de aristas.

        Parameters
        ----------------
            df: pandas.DataFrame
                El DataFrame que contiene la lista de aristas.
            source: str, default="source"
                El nombre de la columna de `df` que contiene los nodos de origen.
            target: str, default="target"
                El nombre de la columna de `df` que contiene los nodos de destino.
            directed: bool, default=True
                Indica si el grafo es dirigido o no.
            weight: str, default=None
                El nombre de la columna de `df` que contiene los pesos de las aristas de existir.
            remove_empty: bool, default=True
                Indica si se deben eliminar las aristas cuyo peso sea menor o igual a 0,
                en caso de tratarse de un grafo con peso.
        Returns
        ----------
            graph_tool.Graph: el grafo creado.
        """
        network = graph_tool.Graph(directed=directed)
        n_vertices = max(df[source_column].max(), df[target_column].max()) + 1
        network.add_vertex(n_vertices)

        if weight_column is not None and weight_column in df.columns:
            if not allow_negative_weights:
                df = df[df[weight_column] > 0]
            weight_prop = network.new_edge_property("double")
            network.add_edge_list(
                df.assign(**{weight_column: df[weight_column].astype(np.float64)})[
                    [source_column, target_column, weight_column]
                ].values,
                eprops=[weight_prop],
            )
            network.edge_properties["edge_weight"] = weight_prop
            # network.shrink_to_fit()
            return network
        else:
            network.add_edge_list(df[[source_column, target_column]].values)
            # network.shrink_to_fit()
            return network

    def add_edge_property(self, series, dtype="float"):
        g = self.network
        series_dict = series.to_dict()

        prop_values = [
            series_dict[
                (
                    self.id_to_label[int(e.source())],
                    self.id_to_label[int(e.target())],
                )
            ]
            for e in g.edges()
        ]

        series_prop = g.new_edge_property(dtype, vals=prop_values)
        g.edge_properties[series.name] = series_prop

    def build_edge_data(self):
        """Actualiza la información de las aristas de la red (`edge_data`) a partir del posicionamiento de los nodos
        (`node_layout`). Si la red todavía no almacena información de las aristas,  crea un objeto `Edge` por arista y los
        almacena en el atributo `edge_data`.
        En caso contrario, actualiza la posición de los vértices de las aristas según el atributo `node_layout`.

        Returns
        ----------
            None
        """
        if self.edge_data is None:
            # first time?
            self.edge_data = []

            for i, e in enumerate(self.network.edges()):
                src_idx = int(e.source())
                dst_idx = int(e.target())
                if src_idx == dst_idx:
                    # no support for self connections yet
                    continue

                src = self.node_layout.get_position(src_idx)
                dst = self.node_layout.get_position(dst_idx)

                edge = Edge(src, dst, src_idx, dst_idx, index=i)
                self.edge_data.append(edge)
        else:
            # update positions only! the rest may have been changed manually.
            for data in self.edge_data:
                src_idx = data.index_pair[0]
                dst_idx = data.index_pair[1]
                src = self.node_layout.get_position(src_idx)
                dst = self.node_layout.get_position(dst_idx)
                data.source = src
                data.target = dst
                data.points = [src, dst]

    def layout_nodes(self, *args, **kwargs):
        """
        Aplica un algoritmo de organización para distribuir los nodos de la red en el plano.

        Parameters
        ----------
        *args : positional arguments
            Argumentos posicionales que se pasarán al método de organización de nodos.
            Los argumentos necesarios dependen del método seleccionado, para más información
            ver la documentación de `LayoutStrategy`.

        **kwargs : keyword arguments
            Argumentos nombrados que se pasarán al método de organización de nodos.
            Los argumentos disponibles dependen del método seleccionado, para más información
            ver la documentación de `LayoutStrategy`.

        Returns
        -------
        None

        Notes
        -------
        La función utiliza un método de organización según el argumento `method` especificado al invocar la función.
        Los métodos disponibles son "force-directed", "precomputed", "geographical", and "radial".

        - Si `method` es "force-directed", se usa ForceDirectedLayout para posicionar los nodos.
        - Si `method` es "precomputed", se usa PrecomputedLayout.
        - Si `method` is "geographical", se usa GeographicalLayout. Este método requiere que se entregue un GeoDataFrame (`geodataframe`)
        y el nombre de una columna en este (`node_column`) para mapear nodos a posiciones en un mapa. Si  no se especifica el nombre de la
        columna, se usa "node_column" por defecto.
        - Si `method` es "radial", se usa RadialLayout.

        Luego de aplicar el método de distribución para posicionar los nodos, se invoca al método  `build_edge_data`
        para actualizar la posición de las aristas.

        Raises
        ------
        NotImplementedError
            Si el método de distribución especificado no está implementado por la librería.

        """

        from .layouts import (
            ForceDirectedLayout,
            GeographicalLayout,
            PrecomputedLayout,
            RadialLayout,
        )

        method = kwargs.pop("method", "force-directed")

        if method == "force-directed":
            self.node_layout = ForceDirectedLayout(self)
        elif method == "precomputed":
            self.node_layout = PrecomputedLayout(self)
        elif method == "geographical":
            geodf = kwargs.pop("geodataframe")
            node_column = kwargs.pop("node_column", None)
            self.node_layout = GeographicalLayout(self, geodf, node_column=node_column)
        elif method == "radial":
            self.node_layout = RadialLayout(self)
        else:
            raise NotImplementedError(f"unknown layout method {method}")

        self.node_layout.layout_nodes(*args, **kwargs)
        self.build_edge_data()

    @property
    def num_vertices(self):
        """
        Retorna la cantidad de vértices en la red.

        Returns
        -------------
        int
        """
        return self.network.num_vertices()

    @property
    def num_edges(self):
        """
        Retorna la cantidad de aristas en la red.

        Returns
        -------------
        int
        """
        return self.network.num_edges()

    @property
    def vertices(self):
        """
        Retorna un iterador que recorre la lista de vértices de la red.
        El orden en el cual se itera sobre los vértices corresponde al orden del índice de los vértices.

        Returns
        -------------
         :meth:`iterator <iterator.__iter__>` 
        """
        return self.network.vertices()

    @property
    def edges(self):
        """
        Retorna un iterador que reccore la lista de aristas de la red.

        Returns
        -------------
         :meth:`iterator <iterator.__iter__>` 
        """
        return self.network.edges()

    @property
    def is_directed(self):
        """
        Indica si el grafo es dirigido o no.

        Returns
        -------------
        Bool
        """
        return self.network.is_directed()

    @property
    def graph(self):
        """
        El grafo que almacena la estructura de red.

        Returns
        ---------
        `graph_tool.Graph <https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.Graph.html>`__
        """
        return self.network

    def shortest_path(self, src, dst, **kwargs):
        """
        Encuentra todos los caminos más cortos entre dos nodos en la red.

        Parameters
        -------------
        src : int
            Identificador del nodo de origen.
        dst : int
            Identificador del nodo de destino.
        *args : argumentos posicionales
            Argumentos posicionales adicionales para pasar al algoritmo de camino más corto.
            Una lista completa de las opciones disponibles se encuentra en la documentación de `graph-tool <https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.topology.all_shortest_paths.html#graph-tool-topology-all-shortest-paths>`__.
        **kwargs : argumentos de palabras clave
            Argumentos de palabras clave adicionales para pasar al algoritmo de camino más corto.
            Una lista completa de las opciones disponibles se encuentra en la documentación de `graph-tool <https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.topology.all_shortest_paths.html#graph-tool-topology-all-shortest-paths>`__.

        Returns
        ----------
        List[List[int]]
            Una lista de caminos más cortos, donde cada camino está representado como una lista de etiquetas de nodos.
            Las etiquetas se obtienen a partir del mapeo `id_to_label`.

        """

        weights = kwargs.pop("weights", self._edge_weight)
        paths = list(
            graph_tool.topology.all_shortest_paths(
                self.network,
                self.node_map[src],
                self.node_map[dst],
                weights=weights,
                **kwargs,
            )
        )

        return [[self.id_to_label[v] for v in path] for path in paths]

    def subgraph(
        self,
        nodes=None,
        vertex_filter=None,
        edge_filter=None,
        keep_positions=True,
        remove_isolated=True,
        copy=False,
    ):
        """
    Crea un subgrafo de la red a partir de un subconjunto de nodos y/o filtros de vértices/aristas.

    Parameters
    ------------
    nodes : list, default=None, optional
        Lista de nodos que se incluirán en el subgrafo. Si se proporciona, el subgrafo contendrá
        solo los nodos especificados.
    vertex_filter : callable or graph_tool.PropertyMap or numpy.ndarray, default=None, optional
        Función de filtro de vértices para seleccionar los vértices que se incluirán en el subgrafo.
        Puede ser tanto un PropertyMap de valores booleanos o numpy.ndarray, que especifican qué vértices se seleccionan,
        o una función que devuelve True si se debe incluir un determinado vértice, o False en caso contrario.
    edge_filter : callable, default=None, optional
        Función de filtro de aristas para seleccionar las aristas que se incluirán en el subgrafo.
        Puede ser tanto un PropertyMap de valores booleanos o numpy.ndarray, que especifican qué aristas se seleccionan,
        o una función que devuelve True si se debe incluir una determinada arista, o False en caso contrario.
    keep_positions : bool, default=True, optional
        Indica si se deben mantener las posiciones de los vértices en el subgrafo resultante.
        Si es True y existe un diseño de nodos (node layout), se copiarán las posiciones correspondientes a los vértices
        en el subgrafo.
    copy : bool, default=False, optional
        Indica si se debe realizar una copia profunda del subgrafo. Si es False, el subgrafo compartirá
        los datos subyacentes con el grafo original.

    Returns
    ---------
    Network
        Un nuevo objeto Network que representa el subgrafo resultante.

    Raises
    ------
    ValueError
        Se produce cuando no se especifica al menos un filtro.

    Notes
    --------
    Para saber más sobre cómo hacer los filtros en base a PropertyMaps, leer la documentación correspondiente de `Graph-Tool <https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.PropertyMap.html#graph_tool.PropertyMap>`__. 

    Examples
    --------
    net = Network()
    # Crear un subgrafo con nodos específicos
    sub = net.subgraph(nodes=[1, 2, 3])
    # Crear un subgrafo utilizando una función de filtro de vértices
    sub = net.subgraph(vertex_filter=lambda x: x in [1, 2, 3])
    # Crear un subgrafo utilizando funciones de filtro de vértices y aristas
    sub = net.subgraph(vertex_filter=dist.ma < 2000, edge_filter=lambda x: x < 10)
    """
        if nodes is not None:
            view = graph_tool.GraphView(
                self.network, vfilt=lambda x: self.id_to_label[x] in nodes
            ).copy()
        elif vertex_filter is not None or edge_filter is not None:
            view = graph_tool.GraphView(
                self.network, vfilt=vertex_filter, efilt=edge_filter
            ).copy()
        else:
            raise ValueError("at least one filter must be specified")

        if remove_isolated:
            degree = view.get_total_degrees(list(view.vertices()))
            view = graph_tool.GraphView(view, vfilt=degree > 0).copy()

        if remove_isolated:
            degree = view.get_total_degrees(list(view.vertices()))
            view = graph_tool.GraphView(view, vfilt=degree > 0).copy()

        old_vertex_ids = set(map(int, view.vertices()))

        if keep_positions and self.node_layout is not None:
            vertex_positions = [
                self.node_layout.get_position(v_id) for v_id in view.vertices()
            ]
        else:
            vertex_positions = None

        node_map_keys = valfilter(lambda x: x in old_vertex_ids, self.node_map).keys()

        view.purge_vertices()
        view.purge_edges()

        if copy:
            view = graph_tool.Graph(view)
            view.shrink_to_fit()

        result = Network(view)
        result.node_map = dict(zip(node_map_keys, map(int, view.vertices())))
        result.id_to_label = itemmap(reversed, result.node_map)

        if vertex_positions:
            result.layout_nodes(method="precomputed", positions=vertex_positions)
        return result

    @property
    def _edge_weight(self):
        """
        Devuelve el peso de las aristas del grafo.

        Returns
        -----------
            PropertyMap or None: El peso de las aristas si está definido, None en caso contrario.
        """
        return (
            self.network.edge_properties["edge_weight"]
            if "edge_weight" in self.network.edge_properties
            else None
        )

    def estimate_node_degree(self, degree_type="in"):
        """
        Calcula el grado de los nodos del grafo. El grado es ponderado según el peso de las aristas, en caso de tener.
        El grado de un nodo indica en cuántas conexiones participa. Si es de entrada, indica cuántas conexiones se dirigen a un nodo;
        si es de salida, cuántas salen desde él.

        Parameters
        ----------
        degree_type : str, default="in", optional
            Tipo de grado a estimar. Puede ser "in" (grado de entrada),
            "out" (grado de salida) o "total" (grado total).

        Returns
        -------
        PropertyMap
            PropertyMap que asigna el grado a cada nodo del grafo.

        Raises
        ------
        ValueError
            Si el tipo de grado especificado no es válido.
        """
        if not degree_type in ("in", "out", "total"):
            raise ValueError("Unsupported node degree")

        vals = getattr(self.network, f"get_{degree_type}_degrees")(
            list(self.network.vertices()),
            eweight=self._edge_weight,
        )

        degree = self.network.new_vertex_property("int", vals=vals)
        self.network.vertex_properties[f"{degree_type}_degree"] = degree

        return degree

    def estimate_betweenness(self, **kwargs):
        """
        Estima la centralidad de intermediación (betweenness centrality) para nodos y aristas en el grafo.
        La centralidad de intermediación es una medida de centralidad que cuantifica la importancia de un nodo o arista basándose
        en su prevalencia en los caminos más cortos entre todos los nodos del grafo.
        Los valores calculados también quedan almacenados en las propiedades de aristas y nodos del grafo asociados a la llave "betweenness".
    
        Parameters
        ------------
        weight : PropertyMap, default=None
            Diccionario con el peso de cada arista
        **kwargs: argumentos de palabras clave
            Argumentos de palabras clave adicionales para configurar el algoritmo.
            Una lista completa de las opciones disponibles se encuentra en la documentación de `graph-tool <https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.centrality.betweenness.html#graph_tool.centrality.betweenness>`__.

        Returns
        -------
        node_centrality : PropertyMap
            Valores de centralidad de intermediación para los nodos.
        edge_centrality : PropertyMap
            Valores de centralidad de intermediación para las aristas.

        """

        weight = kwargs.pop("weight", self._edge_weight)

        node_centrality, edge_centrality = graph_tool.centrality.betweenness(
            self.network, weight=weight, **kwargs
        )

        self.network.edge_properties["betweenness"] = edge_centrality
        self.network.vertex_properties["betweenness"] = node_centrality

        return node_centrality, edge_centrality

    def estimate_pagerank(self, **kwargs):
        """
        Calcula PageRank para cada nodo en la red.
        PageRank asigna una puntuación de importancia a cada nodo en función
        de la estructura de conexión de la red. Cuanto mayor sea la puntuación de PageRank de
        un nodo, más importante se considera en la red.

        Los valores calculados quedan almacenados en las propiedades de nodos del grafo asociados a la llave "pagerank".

        Parameters
        ----------
        weight : PropertyMap, default=None
            Diccionario con el peso de cada arista

        **kwargs: argumentos de palabras clave
            Argumentos de palabras clave adicionales para configurar el algoritmo.
            Una lista completa de las opciones disponibles se encuentra en la documentación de `graph-tool <https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.centrality.pagerank.html>`__.

        Returns
        -------
        node_centrality : PropertyMap
            La puntuación de PageRank para cada nodo en la red.

        """
        weight = kwargs.pop("weight", self._edge_weight)

        node_centrality = graph_tool.centrality.pagerank(
            self.network, weight=weight, **kwargs
        )

        self.network.vertex_properties["pagerank"] = node_centrality
        return node_centrality

    def connected_components(self, directed=True):
        """
        Calcula las componentes conexas de la red y asigna una etiqueta a cada nodo indicando a qué componente pertenece.
        Las componentes conexas son conjuntos de nodos en un grafo donde cada nodo está conectado directa o indirectamente
        con todos los demás nodos de la componente

        Parameters
        ----------
        directed : bool, default=True, opcional
            Indica si se deben considerar las aristas como dirigidas o no dirigidas al calcular las componentes conexas.
            Si el grafo es dirigido, la función encontrará las componentes fuertemente conexas.

        Returns
        -------
        comp : VertexPropertyMap
            PropertyMap con la etiqueta de la componente a la cual pertenece cada vertice.
        histogram : ndarray
            Un array que guarda los tamaños de las componentes conexas, es decir, cuántos nodos contiene.

        """
        return graph_tool.topology.label_components(self.network, directed=directed)

    def largest_connected_component(self, directed=True, copy=False):
        """
        Devuelve la componente conexa más grande del grafo, es decir, la que contiene más vértices.
        Calcula las componentes conexas  del grafo y retorna el subgrafo correspondiente a la componente
        conexa más grande.

        Parameters
        -------------
        directed : bool, default=True, optional
            Indica si el grafo se considera dirigido o no dirigido al calcular las componentes.
        copy : bool, default=False, optional
            Indica si se debe realizar una copia profunda del subgrafo. Si es False, el subgrafo compartirá
            los datos subyacentes con el grafo original.

        Retorna
        -------
        view : Network
            Una instancia de la clase Network que representa el subgrafo correspondiente a la componente conexa más grande.

        """
        components = self.connected_components(directed=directed)
        view = self.subgraph(
            vertex_filter=lambda x: components[0][x] == np.argmax(components[1]),
            copy=copy,
        )
        return view

    def detect_communities(
        self,
        random_state=42,
        method="sbm",
        hierarchical_initial_level=1,
        hierarchical_covariate_type="real-exponential",
        ranked=False,
    ):
        """
        Detecta comunidades de nodos en el grafo utilizando el modelo de detección especificado. 
        Las comunidades son grupos de nodos que estén altamente conectados entre sí en comparación con las conexiones de los demás nodos.

         Si `method` es "sbm", se utilizará el modelo de bloque estocástico para la detección de comunidades. El resultado será almacenado en `self.state`. Para más información acerca del algoritmo usado, ver la `documentación <https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.inference.minimize_blockmodel_dl.html#graph_tool.inference.minimize_blockmodel_dl>`__.

         Si `method` es "hierarchical", se utilizará el modelo jerárquico para la detección de comunidades. Se construirá el árbol de comunidades
         (community_tree) y se almacenará la raíz del árbol (community_root) en `self.community_tree` y `self.community_root` respectivamente. También se calcularán los niveles de comunidades por nodo y se almacenarán en `self.communities_per_level`.
         Para más información acerca del algoritmo usado, referirse a la documentación de `Graph_Tool <https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.inference.minimize_nested_blockmodel_dl.html>`__.

         Si `method` es "ranked", se utilizará el modelo clasificado (ranked) para la detección de comunidades. Para más información acerca del algoritmo usado, referirse a la documentación de `Graph_Tool <https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.inference.minimize_nested_blockmodel_dl.html>`__.

        Después de ejecutar el algoritmo de detección de comunidades correspondiente, se asignarán las etiquetas de comunidad
        a los nodos del grafo y se almacenarán en `self.network.vertex_properties["community"]`.

        El resultado final de la detección de comunidades se almacenará en las siguientes propiedades de la red:

        - self.state: Estado del modelo de detección de comunidades.
        - self.community_tree: Árbol de comunidades (solo si method es "hierarchical").
        - self.community_root: Raíz del árbol de comunidades (solo si method es "hierarchical").
        - self.communities_per_level: Contiene un arreglo por nivel, en el cual se indica la comunidad a la que pertenece cada nodo en el nivel (solo si method es "hierarchical").

        Parameters
        ------------
        random_state : int, default=42
            Semilla utilizada por el generador de números aleatorios para garantizar reproducibilidad.
        method : str, default="sbm"
            Método utilizado para la detección de comunidades. Puede ser "sbm" (Stochastic Block Model),
            "hierarchical" (Modelo Jerárquico) o "ranked" (Modelo Clasificado).
        hierarchical_initial_level : int, default=1
            Nivel inicial para la detección de comunidades jerárquicas. Solo se aplica si `method` es "hierarchical".
        hierarchical_covariate_type : str, default="real-exponential"
            Tipo de covariante utilizada en la detección de comunidades jerárquicas. Solo se aplica si `method` es "hierarchical".
            Puede ser "real-exponential" (real-exponencial) u otro tipo de covariante compatible, para más información
            referirse a la documentacion de `GraphTool <https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.inference.BlockState.html#graph_tool.inference.BlockState>`__.

        """
        np.random.seed(random_state)
        graph_tool.seed_rng(random_state)

        self.communities_per_level = None
        self.community_tree = None
        self.community_root = None

        if method == "sbm":
            self.state = graph_tool.inference.minimize_blockmodel_dl(
                self.network, state_args={"eweight": self._edge_weight}
            )

            self.communities_per_level = None
            self.community_tree = None
            self.community_root = None

            vals = np.array(self.state.get_blocks().a)
        elif method == "hierarchical":
            state_args = dict()

            if self._edge_weight is not None:
                state_args["recs"] = [self._edge_weight]
                state_args["rec_types"] = [hierarchical_covariate_type]

            self.state = graph_tool.inference.minimize_nested_blockmodel_dl(
                self.network, state_args=state_args
            )

            self.community_tree, self.community_root = self._build_community_tree()
            self.communities_per_level = self._build_node_memberships()

            vals = np.array(self.state.get_bs()[hierarchical_initial_level])
        elif method == "ranked":
            state_args = dict()
            state_args["eweight"] = self._edge_weight
            state_args["base_type"] = graph_tool.inference.RankedBlockState
            self.state = graph_tool.inference.minimize_nested_blockmodel_dl(
                self.network, state_args=state_args
            )

            vals = np.array(self.state.levels[0].get_blocks().a)
        else:
            raise ValueError("unsupported method")

        community_prop = self.network.new_vertex_property("int", vals=vals)
        self.network.vertex_properties["community"] = community_prop

    def set_community_level(self, level: int):
        """
        Establece la propiedad de vértices "community" del grafo como el conjunto de comunidades correspondiente
        al nivel de jerarquía especificado. Este método puede usarse si se detectaron las comunidades de nodos usando
        el método jerárquico.
        Esta función se utiliza para definir con qué comunidades se trabajará.
        
        Parameters
        ------------
        level: int
            Nivel dentro de la jerarquía de comunidades con el cual se desea trabajar.
        """
        vals = self.get_community_labels(level)
        community_prop = self.network.new_vertex_property("int", vals=vals)
        self.network.vertex_properties["community"] = community_prop

    def get_community_labels(self, level: int = 0):
        """
        Obtiene las etiquetas correspondientes a las comunidades de nodos existentes en el nivel jerárquico especificado.
        
        Parameters
        ------------
        level: int, default=0
            Nivel dentro de la jerarquía de comunidades.
        
        Raises
        ------
        Exception
            Se produce si no se ha ejecutado la detección de comunidades previamente.

        Returns
        ---------
        numpy.ndarray
            Arreglo numpy que contiene las etiquetas de comunidades del nivel jerárquico especificado.

        """
        if self.communities_per_level is not None:
            return self.communities_per_level[level]

        elif "community" in self.network.vertex_properties:
            return np.array(self.network.vertex_properties["community"].a)

        else:
            raise Exception("must run community detection first")

    def _build_community_tree(self):
        """
        Construye el árbol jerárquico de comunidades detectado al ejecutar el método `detect_communities`.

        Raises
        ------
        Exception
            Se produce si no se ha ejecutado la detección de comunidades previamente.

        Returns
        -------
        tree: graph_tool.Graph
            El árbol de comunidades, en el cual en cada nivel hay un conjunto de comunidades,
            y en el nivel siguiente están las subcomunidades contenidas en las del nivel previo.
        root_idx: int
            El índice del nodo raíz del árbol.

        """

        if self.state is None:
            raise Exception("must detect hierarchical communities first.")

        (
            tree,
            membership,
            order,
        ) = graph_tool.inference.nested_blockmodel.get_hierarchy_tree(
            self.state, empty_branches=False
        )
        self.nested_graph = tree

        new_nodes = 0
        root_node_level = 0
        for i, level in enumerate(self.state.get_bs()):
            level = list(level)
            _nodes = len(np.unique(level))
            if _nodes == 1:
                # new_nodes += 1
                break
            else:
                new_nodes += _nodes
                root_node_level += 1

        root_idx = list(tree.vertices())[self.network.num_vertices() + new_nodes]

        tree.graph_properties["root_node"] = tree.new_graph_property(
            "int", val=root_idx
        )

        return tree, root_idx

    def _build_node_memberships(self):
        """
    Construye un mapeo de pertenencia de nodos a comunidades para cada nivel en la jerarquía de comunidades
    detectadas previamente.

    Returns
    -------
    dict
        Un diccionario que mapea los niveles de la jerarquía de comunidades a un arreglo que lista la comunidad a la que
        pertenece cada nodo en ese nivel. Es importante el orden en el cual se encuentran los ids de comunidad pues la posición
        en el arreglo indica a qué nodo corresponde.

    """
        tree, root = self.community_tree, self.community_root

        depth_edges = graph_tool.search.dfs_iterator(tree, source=root, array=True)

        membership_per_level = defaultdict(lambda: defaultdict(int))

        stack = []
        for src_idx, dst_idx in depth_edges:
            if not stack:
                stack.append(src_idx)

            if dst_idx < self.network.num_vertices():
                # leaf node
                path = [dst_idx]
                path.extend(reversed(stack))

                for level, community_id in enumerate(path):
                    membership_per_level[level][dst_idx] = community_id
            else:
                while src_idx != stack[-1]:
                    # a new community, remove visited branches
                    stack.pop()

                stack.append(dst_idx)

        # removing the lambda enables pickling
        membership_per_level = dict(membership_per_level)
        membership_per_level = valmap(
            lambda x: np.array([x[v_id] for v_id in self.network.vertices()]),
            membership_per_level,
        )

        return membership_per_level
