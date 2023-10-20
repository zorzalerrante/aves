import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
import numpy as np
import seaborn as sns
from cytoolz import unique
from sklearn.preprocessing import minmax_scale

from aves.models.network import Network
from aves.visualization.primitives import RenderStrategy


class NodeStrategy(RenderStrategy):
    """
    Interfaz para las estrategias de renderización de nodos.

    Attributes
    ------------
    network : Network
        La red a visualizar.

    """

    def __init__(self, network: Network, **kwargs):
        """
        Inicializa el objeto NodeStrategy.

        Parameters
        ------------
        network : Network
            La red a visualizar.
        **kwargs : dict, opcional
            Argumentos adicionales (sin uso).
        """
        self.network = network
        super().__init__(network.node_layout.node_positions_vector)

    def prepare_data(self):
        """
        Prepara los datos antes de renderizar los nodos.
        """
        pass

    def name(self):
        """
        Retorna el nombre de la estrategia.

        Returns
        -------
        str
            El nombre de la estrategia de representación de aristas.

        """
        return "node-strategy"


class PlainNodes(NodeStrategy):
    """
    Estrategia para renderizar nodos de un grafo que visualiza estos como puntos en el plano. Permite ajustar el color y tamaño
    de los vértices en la visualización según la categoría o peso de estos.

    Attributes
    -----------
    network : Network
        La red/grafo a visualizar.
    weights : np.array
        Un arreglo que representa los pesos de los nodos. Si no es proporcionado, todos los nodos tendrá el mismo tamaño.
    size : np.array
        Un arreglo que almacena los tamaños de los nodos escalados en base a sus pesos.
    node_categories : str or list, optional
        Categorías de los nodos para colorearlos de acuerdo a estas. Puede ser un nombre de propiedad de vértice (str) o una lista de categorías.
    unique_categories : list, optional
        Lista de categorías únicas para colorear los nodos, si se proporcionan categorías.

    """

    def __init__(self, network: Network, **kwargs):
        """
        Inicializa el objeto PlainNodes.

        Parameters
        ----------
        network : Network
            La red a visualizar.
        weights : str or array-like or None, optional
            Si se provee un string, este debe corresponder a una propiedad de vértice `network` a partir de la cual se calculará el peso.
            
            Si se entrega un arreglo, este representa el peso de los nodos. Esto determina el tamaño de los nodos en la visualización;
            si no es provisto, todos los nodos serán del mismo tamaño.
        node_categories : str or array-like, optional
            Un string o una lista que define las categorías de los nodos. Si es entregado, los nodos serán coloreados 
            según su categoría.

        Raises
        ------
        ValueError
            Si el string entregado como peso no corresponde a una propiedad de vértice válida.
            Si los pesos son provistos pero no son del tipo `numpy array` o no tienen un estructura de arreglo.
    
        """
        super().__init__(network, **kwargs)

        weights = kwargs.get("weights", None)

        if type(weights) == str:
            if not weights in self.network.network.vertex_properties:
                if weights in ("in_degree", "out_degree", "total_degree"):
                    self.network.estimate_node_degree(degree_type=weights.split("_")[0])
                elif weights == "pagerank":
                    self.network.estimate_pagerank()
                elif weights == "betweenness":
                    self.network.estimate_betweenness()
                else:
                    raise Exception("weights must be a valid vertex property if str")

            weights = np.array(self.network.network.vertex_properties[weights].a)

        if weights is not None and not type(weights) in (np.array, np.ndarray):
            raise ValueError(f"weights must be np.array instead of {type(weights)}.")

        self.weights: np.array = weights
        self.size: np.array = None

        self.node_categories = kwargs.pop("categories", None)
        if self.node_categories is not None:
            if type(self.node_categories) == str:
                self.node_categories = list(
                    self.network.network.vertex_properties[self.node_categories].a
                )
            self.unique_categories = sorted(unique(self.node_categories))
        else:
            self.unique_categories = None

    def prepare_data(self):
        """
        Prepara los datos antes de renderizar los vérticas, escalando el tamaño de cada nodo en base a su peso en caso de tener.
        El resultado de esta operación queda almacenado en `size`.
        """
        if self.weights is not None:
            self.size = minmax_scale(np.sqrt(self.weights), feature_range=(0.01, 1.0))

    def render(self, ax, *args, **kwargs):
        """
        Renderiza los nodos en una visualización.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Los ejes donde se dibujarán los nodos.
        node_size : int o float, default=10 optional
            El tamaño de los nodos en el gráfico.
        palette : str o lista de colores, default=plasma, optional
            La paleta de colores utilizada para mapear las categorías de los nodos a colores.
    
        Raises
        ------
        ValueError
            Si el númer de colores en la paleta no coincide con la cantidad de categorías.
            Si la paleta no tiene un nombre válido (según la librería Seaborn) o no corresponde a un iterable de colores.
            
        """
        node_size = kwargs.pop("node_size", 10)

        if self.node_categories is not None:
            palette_name = kwargs.pop("palette", "plasma")

            if isinstance(palette_name, str):
                palette = sns.color_palette(
                    palette_name, n_colors=len(self.unique_categories)
                )
            elif palette_name is not None:
                # assume it's an iterable of colors
                palette = list(palette_name)
                if len(palette) != len(self.unique_categories):
                    raise ValueError(
                        "the number of colors does not match the number of categories"
                    )
            else:
                raise ValueError(
                    "palette must be a valid name or an iterable of colors"
                )
            color_map = dict(zip(self.unique_categories, palette))

            c = [color_map[c] for c in self.node_categories]
        else:
            kwargs.pop("palette", "plasma")
            c = None

        if self.weights is None:
            ax.scatter(
                self.data[:, 0], self.data[:, 1], *args, s=node_size, c=c, **kwargs
            )
        else:
            ax.scatter(
                self.data[:, 0],
                self.data[:, 1],
                *args,
                s=self.size * node_size,
                c=c,
                **kwargs,
            )

    def name(self):
        return "plain"


class LabeledNodes(PlainNodes):
    """
    Estrategia para renderizar nodos que los representa como puntos en el plano, junto a su etiqueta.
    Esta clase hereda de PlainNodes y añade la capacidad de agregar etiquetas de texto a los nodos en el gráfico.

    Attributes
    ------------
    network : Network
        La red a visualizar.
    label_property : str
        El nombre de la propiedad de vértice que contiene las etiquetas de texto para los nodos, en `network`.
    labels : list(tuple)
        Una lista que contiene las etiquetas de texto junto a sus coordenadas y cofiguración para cada nodo en el gráfico.
    radial : bool, optional
        Un valor booleano que indica si las etiquetas se deben representar radialmente alrededor del nodo.
    offset : float, optional
        Un valor que controla la distancia radial de las etiquetas desde los nodos si `radial` es True,

    """
    def __init__(self, network: Network, label_property, **kwargs):
        """
        Parameters
        ----------
         network : Network
            La red a visualizar.
        label_property : str
            El nombre de la propiedad de vértice que contiene las etiquetas de texto para los nodos, en `network`.
        labels : list
            Una lista que contiene las coordenadas, las etiquetas de texto y los argumentos de texto para cada nodo en el gráfico.
        radial : bool, default=False, optional
            Indica si las etiquetas se deben representar radialmente alrededor del nodo.
        offset : float, default=0.0, optional
            Controla la distancia radial de las etiquetas desde los nodos si `radial` es True,
        """
        super().__init__(network, **kwargs)
        self.labels = []
        self.label_property = label_property

        self.radial = kwargs.get("radial", False)
        self.offset = kwargs.get("offset", 0.0)

    def prepare_data(self):
        """
        Prepara los datos antes de representar los nodos. Llama al método de la clase base `PlainNodes` y
        luego para cada nodo agrega su etiqueta, su posición en el plano y la configuración del texto al atributo `labels`.
        """
        super().prepare_data()

        graph = self.network.graph

        for idx in graph.vertices():
            label = graph.vertex_properties[self.label_property][idx]
            if label:
                label = str(label)
                text_args = {}

                if self.radial:
                    degrees = self.network.node_layout.get_angle(idx)
                    ratio = self.network.node_layout.get_ratio(idx) + self.offset
                    radians = np.radians(degrees)

                    pos = np.array([ratio * np.cos(radians), ratio * np.sin(radians)])

                    text_args["ha"] = "left" if pos[0] >= 0 else "right"
                    text_rotation = degrees
                    if text_rotation > 90:
                        text_rotation = text_rotation - 180
                    elif text_rotation < -90:
                        text_rotation = text_rotation + 180

                    text_args["rotation"] = text_rotation

                    text_args["rotation_mode"] = "anchor"
                else:
                    pos = self.data[int(idx)]
                    text_args["ha"] = "center"

                text_args["va"] = "center"

                self.labels.append((pos, label, text_args))

    def render(self, ax, *args, **kwargs):
        """
        Renderiza los nodos en el plano especificado, junto a sus etiquetas.

        Parameters
        ------------
        ax : matplotlib.axes.Axes
            Los ejes donde se dibujarán los nodos.
        node_size : int o float, default=10 optional
            El tamaño de los nodos en el gráfico.
        palette : str o lista de colores, default=plasma, optional
            La paleta de colores utilizada para mapear las categorías de los nodos a colores.
        fontsize : str o int, default="medium", optional
            El tamaño de fuente para las etiquetas de texto.
        text_color : str, default="white", optional
            El color del texto.
        """
        fontsize = kwargs.pop("fontsize", "medium")
        text_color = kwargs.pop("text_color", "white")

        super().render(ax, *args, **kwargs)

        for pos, label, text_args in self.labels:
            text_args["fontsize"] = fontsize
            text_args["color"] = text_color

            text = ax.text(pos[0], pos[1], label, **text_args)
            text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=1, foreground="black"),
                    path_effects.Normal(),
                ]
            )

    def name(self):
        return "labeled"
