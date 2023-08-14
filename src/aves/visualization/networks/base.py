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
        plotea
        """
        nodes = kwargs.get("nodes", {})
        edges = kwargs.get("edges", {})
        zorder = kwargs.get("zorder", None)
        self.plot_edges(ax, zorder=zorder, **edges)
        self.plot_nodes(ax, zorder=zorder, **nodes)

    def bundle_edges(self, method: str, *args, **kwargs):
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
        if method == "weighted":
            self.edge_strategy = WeightedEdges(
                self.network,
                kwargs.get("weights", "edge_weight"),
                kwargs.get("k", 5),
                kwargs.get("scheme", "bins"),
                kwargs.get('bins', None)
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
            self.edge_strategy = PlainEdges(self.network)
        else:
            raise ValueError(f"{method} is not supported")

        self.edge_strategy_args = dict(**kwargs)

    def set_node_drawing(self, method="plain", **kwargs):
        if method == "plain":
            self.node_strategy = PlainNodes(self.network, **kwargs)
        elif method == "labeled":
            labels = kwargs.get("label", "elem_id")
            self.node_strategy = LabeledNodes(self.network, labels, **kwargs)
        else:
            raise ValueError(f"{method} is not supported")

        self.node_strategy_args = dict(**kwargs)

    def set_node_labels(self, func=None):
        graph = self.network.graph
        self.labels = graph.new_vertex_property("string")

        for idx in graph.vertices():
            if func is not None:
                self.labels[idx] = func(self.network, idx)
            else:
                self.labels[idx] = f"{self.network.id_to_label[int(idx)]}"

        self._maybe_update_strategies(edges=False)
