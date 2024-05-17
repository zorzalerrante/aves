import graph_tool
import graph_tool.draw
import graph_tool.inference
import graph_tool.topology

from aves.models.network import Network

from .edges import *
from .fdeb import FDB
from .heb import HierarchicalEdgeBundling
from .nodes import *


class NodeLink(object):
    """
    La visualización de un grafo tipo `Network`.

    Esta clase permite crear visualizaciones de la red utilizando diferentes estrategias de diseño.

    Attributes
    ----------
    network : Network
        El objeto Network que representa la red.
    bundle_model : NoneType
        El modelo de agrupamiento de aristas.
    edge_strategy : EdgeStrategy, optional
        La estrategia utilizada para dibujar las aristas.
    edge_strategy_args : dict, optional
        Los argumentos adicionales usados para dibujar las aristas.
    node_strategy : NodeStrategy, optional
        La estrategia utilizada para dibujar los nodos.
    node_strategy_args : dict, optional
        Los argumentos adicionales usados al dibujar los nodos.
    labels : NoneType
        Las etiquetas utilizadas en la visualización.

    """

    def __init__(self, network: Network):
        """
        Constructor de la clase.

        Parameters
        ----------
        network : Network
            El objeto Network que representa la red a visualizar.
        """
        self.network = network
        self.bundle_model = None
        self.edge_strategy: EdgeStrategy = None
        self.edge_strategy_args: dict = None

        self.node_strategy: NodeStrategy = None
        self.node_strategy_args: dict = None

        self.labels = None

    def layout_nodes(self, *args, **kwargs):
        """
        Realiza el posicionamiento de los nodos en la visualización y actualiza las estrategias de dibujo si es necesario.

        Parameters
        ----------
        method : string
            Indica el nombre de la heurística de organización de nodos a utilizar en la visualización.
            Puede ser "force-directed", "precomputed", "geographical" o "radial".
        *args : tuple
            Argumentos posicionales pasados al método de diseño de nodos de la red.
        **kwargs : dict
            Argumentos clave pasados al método de diseño de nodos de la red. Los parámetros necesarios
            dependerán del método que se desea utilizar, por lo que revisa la documentación de :func:`~aves.models.network.base.Network.layout_nodes`
            para saber qué argumentos usar.

        """
        self.network.layout_nodes(*args, **kwargs)
        self._maybe_update_strategies()

    def _maybe_update_strategies(self, nodes=True, edges=True):
        """
        Actualiza las estrategias de dibujo de los elementos del grafo si es necesario.

        Parameters
        ----------
        nodes : bool, default=True, optional
            Indica si se deben actualizar las estrategias de dibujo de nodos.
        edges : bool, default=True, optional
            Indica si se deben actualizar las estrategias de dibujo de aristas.

        """
        if edges and self.edge_strategy is not None:
            self.set_edge_drawing(
                method=self.edge_strategy.name(), **self.edge_strategy_args
            )

        if nodes and self.node_strategy is not None:
            self.set_node_drawing(
                method=self.node_strategy.name(), **self.node_strategy_args
            )

    def plot_edges(self, ax, *args, **kwargs):
        """
        Dibuja las aristas de la red utilizando la estrategia de dibujo previamente escogida. Si `edge_estrategy` es
        igual a `None` al momento de ejecutar el método, entonces se utilizará la estrategia `plain`.

        Parameters
        ----------
        ax : Axes
            El plano en el cual se dibujarán las aristas.
        *args : tuple
            Argumentos posicionales adicionales que serán pasados a la función :func:`~aves.visualization.networks.edges.PlainEdges.render` .
            Una lista de los argumentos a usar se encuentra en la documentación correspondiente a la estrategia usada.
        **kwargs : dict
            Argumentos clave adicionales que serán pasados a la función :func:`~aves.visualization.networks.edges.PlainEdges.render` .
            Una lista de los argumentos a usar se encuentra en la documentación correspondiente a la estrategia usada.

        """

        if self.edge_strategy is None:
            self.set_edge_drawing()

        self.edge_strategy.plot(ax, *args, **kwargs)

    def plot_nodes(self, ax, *args, **kwargs):
        """
        Dibuja los nodos de la red utilizando la estrategia de dibujo previamente escogida. Si `node_estrategy` es
        igual a `None` al momento de ejecutar el método, entonces se utilizará la estrategia `plain`.

        Parameters
        ----------
        ax : Axes
            El plano en el cual se dibujarán los nodos.
        *args : tuple
            Argumentos posicionales adicionales que serán pasados a la función :func:`~aves.visualization.networks.edges.PlainNodes.render` .
            Una lista de los argumentos a usar se encuentra en la documentación correspondiente a la estrategia usada.
        **kwargs : dict
            Argumentos clave adicionales que serán pasados a la función :func:`~aves.visualization.networks.edges.PlainNodes.render` .
            Una lista de los argumentos a usar se encuentra en la documentación correspondiente a la estrategia usada.

        """
        if self.node_strategy is None:
            self.set_node_drawing()

        self.node_strategy.plot(ax, *args, **kwargs)

    def plot(self, ax, *args, **kwargs):
        """
        Dibuja una red.

        Puedes encontrar ejemplos de uso de esta función en el notebook  `notebooks/vis-course/04-python-redes-preliminario.ipynb`.

        Parameters
        ----------
        ax : Axes
            El plano en el cual se dibujará la red.
        **kwargs : dict
            Argumentos clave adicionales pasados a los métodos de dibujo de nodos y aristas.
                - nodes : dict, optional
                    Un diccionario que contiene argumentos que serán pasados al método :func:`~aves.visualization.networks.base.NodeLink.plot_nodes`  al dibujar los nodos.
                    Esto permite personalizar la apariencia y el comportamiento del proceso de dibujo de los nodos.
                - edges : dict, optional
                    Un diccionario que contiene argumentos que serán pasados al método :func:`~aves.visualization.networks.base.NodeLink.plot_edges`  al dibujar las aristas.
                    Esto permite personalizar la apariencia y el comportamiento del proceso de dibujo de las aristas.
                - zorder : int, optional
                    Z-order para el dibujo de los elementos. Esto indica la jerarquia visual en la cual mostrar los elementos en el plano, los elementos
                    con mayor z-order aparecerán "encima" de aquellos con menor z-order.

        """
        nodes = kwargs.get("nodes", {})
        edges = kwargs.get("edges", {})
        zorder = kwargs.get("zorder", None)
        self.plot_edges(ax, zorder=zorder, **edges)
        self.plot_nodes(ax, zorder=zorder, **nodes)

    def bundle_edges(self, method: str, *args, **kwargs):
        """
        Agrupa las aristas en la visualización de la red usando el método especificado.
        El agrupamiento de aristas combina aristas cercanas en la visualización
        de una red para mejorar la claridad y reducir la sobrecarga visual.

        Parameters
        ------------
        method : str
            El método a usar. Actualmente están implementados:
            - "force-directed": Agrupar las aristas usando el método `force directed`.
            - "hierarchical": Agrupar las aristas usando el método jerárquico.

        tree: graph_tool.GraphView, optional
            Solo se usa en el método jerárquico. Corresponde el árbol de jerarquia de comunidades de la red.
            Si no se provee, se obtendrá de :attr:`~aves.visualization.networks.base.NodeLink.network`.

        root_vertex: int
            Solo se usa en el método jerárquico. Corresponde a la raíz del árbol de jerarquía.
            Si no se provee, se obtendrá de :attr:`~aves.visualization.networks.base.NodeLink.network`.

        *args, **kwargs
            Argumentos adicionales para configurar la agrupación, esto dependerá del método especificado.
            Para más información, ver la documentación de :class:`~aves.visualization.networks.fdeb.FDB` o
            :attr:`~aves.visualization.networks.heb.HierarchicalEdgeBundling`.

        Returns
        -------
        bundle_model : HierarchicalEdgeBundling o FDB
            El modelo de agrupamiento de aristas creado según el método escogido.

        Raises
        ------
        ValueError
            Si el método especificado no está implementado.

        """
        if method == "force-directed":
            self.bundle_model = FDB(self.network, *args, **kwargs)
        elif method == "hierarchical":
            tree = kwargs.get("tree", None)
            if tree is None:
                tree, root = self.network.community_tree, self.network.community_root
            else:
                root = kwargs.get("root_vertex")
            self.bundle_model = HierarchicalEdgeBundling(
                self.network, tree, root, *args, **kwargs
            )
        else:
            raise ValueError(f"method {str} not supported")

        self._maybe_update_strategies()

        return self.bundle_model

    def set_edge_drawing(self, method="plain", **kwargs):
        """
        Configura la estrategia de dibujo de aristas para la visualización de la red. Este método modifica los
        atributos :attr:`~aves.visualization.networks.base.NodeLink.edge_strategy` y :attr:`~aves.visualization.networks.base.NodeLink.edge_strategy_args`.

        Parameters
        ----------
        method : str, default="plain", optional
            El método de visualización de aristas.
            Actualmente están implementados: "weighted", "origin-destination", "community-gradient", "plain".
        **kwargs : dict
            Argumentos adicionales específicos para cada método de dibujo.
            Para ver las opciones disponibles referirse a la documentación de :class:`~aves.visualization.networks.edges.EdgeStrategy`.

        Raises
        ------
        ValueError
           Si el método especificado no está implementado o si hay un problema al ejecutar la estrategia seleccionada.

        Returns
        -------
        None
        """

        curved_edges = kwargs.get("curved", False)

        if method == "weighted":
            self.edge_strategy = WeightedEdges(
                self.network,
                kwargs.get("weights", "edge_weight"),
                kwargs.get("k", 5),
                kwargs.get("scheme", "bins"),
                kwargs.get("bins", None),
                curved_edges,
            )
        elif method == "origin-destination":
            if not self.network.is_directed:
                raise ValueError("method only works with directed graphs")
            self.edge_strategy = ODGradient(self.network, kwargs.get("n_points", 30))
        elif method == "community-gradient":
            if type(self.bundle_model) != HierarchicalEdgeBundling:
                raise ValueError(f"{method} only works with HierarchicalEdgeBundling")
            level = kwargs.get("level", 0)
            communities = self.network.get_community_labels(level)
            self.edge_strategy = CommunityGradient(
                self.network, node_communities=communities
            )
        elif method == "plain":
            self.edge_strategy = PlainEdges(self.network, curved_edges)
        else:
            raise ValueError(f"{method} is not supported")

        self.edge_strategy_args = dict(**kwargs)

    def set_node_drawing(self, method="plain", **kwargs):
        """
        Configura la estrategia de dibujo de nodos para la visualización de la red. Este método modifica los
        atributos :attr:`~aves.visualization.networks.base.NodeLink.node_strategy` y :attr:`~aves.visualization.networks.base.NodeLink.node_strategy_args`.

        Parameters
        ----------
        method : str, default="plain", optional
            El método de visualización de nodos.
            Actualmente están implementados:  "plain", "labeled".
        **kwargs : dict
            Argumentos adicionales específicos para cada método de dibujo.
            Para ver las opciones disponibles referirse a la documentación de :class:`~aves.visualization.networks.nodes.NodeStrategy`.

        Raises
        ------
        ValueError
           Si el método especificado no está implementado o si hay un problema al ejecutar la estrategia seleccionada.

        Returns
        -------
        None
        """
        if method == "plain":
            self.node_strategy = PlainNodes(self.network, **kwargs)
        elif method == "labeled":
            labels = kwargs.get("label", "elem_id")
            self.node_strategy = LabeledNodes(self.network, labels, **kwargs)
        else:
            raise ValueError(f"{method} is not supported")

        self.node_strategy_args = dict(**kwargs)

    def set_node_labels(self, func=None):
        """
        Configura las etiquetas de los nodos en la visualización de la red.

        Parameters
        ----------
        func : function, default=None, optional
            Una función que mapea el índice de un nodo a su etiqueta.
            Si no es provista, se usará el diccionario :attr:`~aves.models.network.base.Network.id_to_label` .

        Returns
        -------
        None
        """
        graph = self.network.graph
        self.labels = graph.new_vertex_property("string")

        for idx in graph.vertices():
            if func is not None:
                self.labels[idx] = func(self.network, idx)
            else:
                self.labels[idx] = f"{self.network.id_to_label[int(idx)]}"

        self._maybe_update_strategies(edges=False)
