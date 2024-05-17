from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from cytoolz import unique
from matplotlib.collections import LineCollection

from aves.models.network import Network
from aves.visualization.collections import ColoredCurveCollection
from aves.visualization.primitives import RenderStrategy
from aves.visualization.colors.palettes import build_palette

from .bezier_edges import curved_edges


class EdgeStrategy(RenderStrategy):
    """Una interfaz para los métodos de visualziación de aristas."""

    def __init__(self, network: Network, **kwargs):
        """
        Inicializa una estrategia de representación de aristas.

        Parameters
        ----------
        network : Network
            La red sobre la cual se aplicará el método de representación de aristas.
        **kwargs : dict
            Argumentos adicionales (sin uso).

        """
        self.network = network
        self.curved_lines = kwargs.get("curved", False)
        super().__init__(self.network.edge_data)

    def name(self):
        """
        Retorna el nombre de la estrategia.

        Returns
        -------
        str
            El nombre de la estrategia de representación de aristas.

        """
        return "strategy-name"

    def build_lines(self, edge_data, bezier_args=None):
        if self.curved_lines:
            if bezier_args is None:
                bezier_args = {}
            lines = curved_edges(edge_data, **bezier_args)
        else:
            lines = []
            for edge in edge_data:
                lines.append(edge.points)

            lines = np.array(lines)

        return lines


class PlainEdges(EdgeStrategy):
    """Estrategia para renderizar las aristas del grafo en la cual las aristas tienen un aspecto uniforme."""

    def __init__(self, network, curved, **kwargs):
        """
        Inicializa una estrategia de visualización de aristas con un único color y aspecto uniforme.

        Parameters
        ----------
        network : Network
            La red sobre la cual se aplicará el método de representación de aristas.
        **kwargs : dict
            Argumentos adicionales (sin uso).

        """
        super().__init__(network, curved=curved)
        self.lines = None

    def prepare_data(self):
        """
        Prepara los datos de las aristas para su visualización.

        Extrae las coordenadas de las aristas y las almacena en el atributo `lines`.

        Returns
        -------
        None
        """
        self.lines = self.build_lines(self.data)

    def render(
        self,
        ax,
        color="#abacab",
        linewidth=1.0,
        linestyle="solid",
        alpha=0.75,
        **kwargs,
    ):
        """
        Agrega las aristas a la visualización de la red.

        Parameters
        ----------
        ax : Matplotlib Axes
            Los ejes en los cuales se dibujará la red.
        color : str, default= "#abacab", optional
            El color de las aristas. Puede indicarse por su representación hexadecimal o nombre.
        linewidth : float, default=1.0, optional
            El grosor de las aristas.
        linestyle : str, default="solid", optional
            El estilo de línea de las aristas.
        alpha : float, default=0.75, optional
            La transparencia de las aristas.
        **kwargs : dict
            Argumentos adicionales que permiten personalizar la visualización. Una lista completa
            de las opciones disponibles se encuentra en la documentación de la librería `Matplotlib <https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.LineCollection>`__.

        Returns
        -------
        LineCollection
            Una colección de líneas que representa las aristas en el gráfico.

        """
        collection = LineCollection(
            self.lines,
            color=color,
            linewidths=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            **kwargs,
        )
        return ax.add_collection(collection)

    def name(self):
        return "plain"


class WeightedEdges(EdgeStrategy):
    """
    Método para renderizar aristas del grafo en el cual las aristas tienen un peso asociado.
    El color de cada arista es determinado por su peso.

    Attributes
    ------------
    network : Network
        Objeto Network que representa la red.
    k : int
        El número de grupos o bins para categorizar las aristas según sus pesos.
    bins : np.array o None
        Los límites de los bins para los pesos categorizados.
    weights : np.array o None
        Los pesos de las aristas a utilizar para el renderizado.
    lines : np.array o None
        Las coordenadas de las aristas a renderizar.

    """

    def __init__(self, network, weights, k, scheme, bins, curved, **kwargs):
        """
        Inicializa el objeto WeightedEdges.

        Parameters
        ----------
        network : Network
            Objeto Network que representa la red.
        weights : str o np.array o None, default=None, optional
            Los pesos de las aristas a utilizar para el renderizado. Si es str, especifica el nombre de la propiedad
            de la arista que contiene los pesos. Si es np.array, proporciona los pesos directamente. Si es None, no se utilizan pesos.
        k : int
            El número de grupos o bins para categorizar las aristas según sus pesos.
        scheme: string
            Estrategia para categorizar las aristas. Puede ser por número de grupos, por cuantiles o personalizada.
        bins: int, sequence of scalars, o IntervalIndex
            Los límites de los rangos de las categorías en caso de que `scheme` sea `custom`.

        **kwargs : dict, opcional
            Argumentos adicionales. (sin uso)

        Raises
        --------
        ValueErrror
            Si `scheme` no corresponde a los valores aceptados.
            Si la configuración personalizada resulta en un grupo con menos de dos aristas.
        """
        super().__init__(network, curved=curved)
        # self.edge_data_per_group = {i: [] for i in range(k)}
        # self.strategy_per_group = {
        #    i: PlainEdges(self.edge_data_per_group[i]) for i in range(k)
        # }
        self.k = k
        self.bins = None
        self.weights = weights
        self.lines = None
        self.scheme = scheme
        if not self.scheme in ("bins", "quantiles", "custom"):
            raise ValueError("scheme must be bins, quantiles or custom")

        if self.scheme == "custom":
            self.bins = np.array(bins)
            if len(bins) < 2:
                raise ValueError("bins must have at least two elements")
            self.bins = bins
            self.k = len(self.bins) - 1
            # print(bins)

    def prepare_data(self):
        """
        Prepara los datos para renderizar las aristas. Almacena las coordenadas de las aristas en el atributo `lines` y divide las aristas
        en `k`  grupos según su peso.

        Raises
        ------
        ValueError
            Si el atributo `weights` no corresponde a una propiedad de las aristas del grafo (en caso de ser un string)
            o si no se almacena como un np.array.
        """
        self.lines = self.build_lines(self.data)

        # for edge in self.data:
        #    self.lines.append(edge.points)

        # self.lines: np.array = np.array(self.lines)

        weights = self.weights

        if type(weights) == str:
            if not weights in self.network.network.edge_properties:
                if weights == "betweenness":
                    self.network.estimate_betweenness()
                else:
                    raise Exception("weights must be a valid edge property if str")

            weights = np.array(self.network.network.edge_properties[weights].a)

        if weights is not None and not type(weights) in (np.array, np.ndarray):
            raise ValueError(f"weights must be np.array instead of {type(weights)}.")

        # weights: np.array = weights
        print(self.scheme)

        if self.scheme == "bins":
            groups, bins = pd.cut(
                weights, self.k, labels=False, retbins=True, duplicates="raise"
            )
            self.bins = bins
        elif self.scheme == "quantiles":
            groups, bins = pd.qcut(
                weights, self.k, labels=False, retbins=True, duplicates="raise"
            )
            self.bins = bins
        else:
            groups = pd.cut(
                weights, bins=self.bins, labels=False, retbins=False, duplicates="raise"
            )
        self.line_groups = groups

        print(self.lines.shape, weights.shape)
        print(self.line_groups)

    def render(self, ax, *args, **kwargs):
        """
        Renderiza las aristas en la visualización de la red

        Parameters
        ------------
        ax : Axes
            El eje en el cual se dibujará el grafo.
        palette: str, optional
            Nombre de la paleta de colores a usar.
        color: str
            Color a partir del cual crear la paleta de colores.
        **kwargs : optional
            Argumentos adicionales que permiten personalizar la visualización. Una lista completa
            de las opciones disponibles se encuentra en la documentación de la librería `Matplotlib <https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.LineCollection>`__.

        Returns
        -------
        list
            Una lista de objetos LineCollection que representan las aristas renderizadas para cada grupo.
        """

        if "color" in kwargs:
            palette = kwargs.pop("color")
        else:
            palette = kwargs.pop("palette", "#a7a7a7")

        edge_colors = build_palette(
            self.bins,
            palette=palette,
            palette_type=kwargs.pop("palette_type", "dark"),
        )

        results = []
        for i in range(self.k):
            coll_lines = self.lines[self.line_groups == i]
            print(len(self.lines), len(self.line_groups), len(coll_lines))

            coll = LineCollection(
                coll_lines,
                color=edge_colors[i],
                **kwargs,
            )
            results.append(ax.add_collection(coll))

        return results

    def name(self):
        return "weighted"


class CommunityGradient(EdgeStrategy):
    """
    Método para renderizar aristas de una red que tiene comunidades de nodos. Cada extremo de una arista es
    coloreado según la comunidad del nodo del cual entra o sale.

    Parameters
    ----------
    network : Network
        Objeto Network que representa la red.
    node_communities : np.array
        Un array numpy que indica a qué comunidad pertenece cada nodo.

    Attributes
    ------------
    network : Network
        Objeto Network que representa la red.
    node_communities : np.array
        Un array numpy que indica a qué comunidad pertenece cada nodo.
    community_ids : list
        Una lista de identificadores únicos de las comunidades.
    community_links : defaultdict
        Un diccionario que mapea pares de comunidades a objetos ColoredCurveCollection, que almacenan las aristas entre esas comunidades.
    """

    def __init__(self, network, node_communities, **kwargs):
        """
        Inicializa el objeto CommunityGradient.

        Parameters
        ----------
        network : Network
            Objeto Network que representa la red.
        node_communities : np.array
            Un array numpy que indica a qué comunidad pertenece cada nodo.
        **kwargs : dict, opcional
            Argumentos adicionales (sin uso).
        """
        super().__init__(network)
        self.node_communities = node_communities
        self.community_ids = sorted(unique(node_communities))
        self.community_links = defaultdict(ColoredCurveCollection)

    def prepare_data(self):
        """
        Prepara los datos para renderizar las aristas, identificando la comunidad de cada nodo participante de la arista. Almacena
        las líneas a trazar en el atributo `community_links`.
        """
        for edge_data in self.data:
            pair = (
                self.node_communities[int(edge_data.index_pair[0])],
                self.node_communities[int(edge_data.index_pair[1])],
            )

            # TODO: add weight
            self.community_links[pair].add_curve(edge_data.points, 1)

    def render(self, ax, *args, **kwargs):
        """
        Renderiza las aristas de la red en una visualización.

        Parameters
        ------------
        ax : Axes
            Los ejes Matplotlib donde se dibujarán las aristas.
        *args : optional
            Argumentos posicionales adicionales para configurar la visualización. Una lista completa se encuentra en la documentación
            de `Matplotlib <https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.LineCollection>`__.
        palette : string, default="plasma", optional
            Paleta de colores a usar en el trazado de las aristas.
        **kwargs : optional
            Argumentos adicionales para configurar la visualización. Una lista completa se encuentra en la documentación
            de `Matplotlib <https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.LineCollection>`__.
        """
        community_colors = dict(
            zip(
                self.community_ids,
                sns.color_palette(
                    kwargs.pop("palette", "plasma"), n_colors=len(self.community_ids)
                ),
            )
        )

        for pair, colored_lines in self.community_links.items():
            colored_lines.set_colors(
                source=community_colors[pair[0]], target=community_colors[pair[1]]
            )
            colored_lines.render(ax, *args, **kwargs)

    def name(self):
        return "community-gradient"


class ODGradient(EdgeStrategy):
    """
    Estrategia para renderizar aristas de una red en la cual el color de cada extremo de la arista
    indica si el nodo correspondiente es el origen o el destino de la arista. Este método solo funciona
    en visualizaciones de grafos dirigidos.

    Attributes
    ------------
    network : Network
        La red a visualizar.
    n_points : int
        Número de puntos para interpolar las aristas al colorearlas.
    colored_curves : ColoredCurveCollection
        Colección de líneas que almacena las aristas.
    """

    def __init__(self, network, n_points, **kwargs):
        """
        Inicializa el objeto ODGradient.

        Parameters
        ------------
        network : Network
            La red a visualizar.
        n_points : int
            Número de puntos para interpolar en las aristas al colorearlas.
        **kwargs : dict, optional
            Argumentos adicionales (sin uso).
        """
        super().__init__(network)
        self.n_points = n_points
        self.colored_curves = ColoredCurveCollection()

    def prepare_data(self):
        """
        Prepara los datos para renderizar las aristas. Almacena las líneas a trazar en el atributo `colored_curves`.
        """
        interp = np.linspace(0, 1, num=self.n_points, endpoint=True)

        for edge_data in self.data:
            if type(edge_data.points) == list and len(edge_data.points) == 2:
                points = np.array(
                    [
                        (edge_data.source * (1 - t) + edge_data.target * t)
                        for t in interp
                    ]
                )
            elif type(edge_data.points) == np.array and edge_data.points.shape[0] == 2:
                points = np.array(
                    [
                        (edge_data.source * (1 - t) + edge_data.target * t)
                        for t in interp
                    ]
                )
            else:
                points = edge_data.points

            # TODO: add weight
            self.colored_curves.add_curve(points, 1)

    def render(self, ax, *args, **kwargs):
        """
        Renderiza las aristas de la red en una visualización.

        Parameters
        ------------
        ax : Axes
            Los ejes Matplotlib donde se dibujarán las aristas.
        *args : optional
            Argumentos posicionales adicionales para configurar la visualización. Una lista completa se encuentra en la documentación
            de `Matplotlib <https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.LineCollection>`__.
        source_color : string, default="blue", optional
            Color que se usará para el extremo "origen" de la arista.
        target_color : string, default="red", optional
            Color que se usará para el extremo "destino" de la arista.
        **kwargs : optional
            Argumentos adicionales para configurar la visualización. Una lista completa se encuentra en la documentación
            de `Matplotlib <https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.LineCollection>`__.
        """

        self.colored_curves.set_colors(
            source=kwargs.pop("source_color", "blue"),
            target=kwargs.pop("target_color", "red"),
        )
        self.colored_curves.render(ax, *args, **kwargs)

    def name(self):
        return "origin-destination"
