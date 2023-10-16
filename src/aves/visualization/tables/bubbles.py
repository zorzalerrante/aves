import pymunk
from pymunk.vec2d import Vec2d
import pymunk.matplotlib_util
import numpy as np

from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import seaborn as sns

import matplotlib.patches as mpatches


def arc_patch(center, radius, theta1, theta2, resolution=50, **kwargs):
    """
    Genera un segmento circular insertable en figuras de Matplotlib. El arco se define por su centro, radio, ángulo inicial
    y ángulo final. El número de puntos utilizados para aproximar el arco se puede ajustar mediante el parámetro de
    resolución.

    Parameters
    ----------
    center : tuple
        Coordenadas (x, y) del centro del arco.
    radius : float
        Radio del arco.
    theta1 : float
        Ángulo inicial del arco en grados.
    theta2 : float
        Ángulo final del arco en grados.
    resolution : int, default=50, opcional
        Número de puntos que se utilizarán para aproximar el arco.
    **kwargs : dict
        Argumentos adicionales para personalizar el polígono. 
        Una lista completa de todas las posibles especificaciones se encuentra en la documentación de `Matplotlib <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html>`__

    Returns
    -------
    mpatches.Polygon
        Patch de matplotlib que representa el arco especificado y se puede insertar en una figura.

    """
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    points = np.vstack(
        (radius * np.cos(theta) + center[0], radius * np.sin(theta) + center[1])
    )
    poly = mpatches.Polygon(points.T, closed=True, **kwargs)
    return poly


def build_dual_bubble(pos=[0, 0], left_percentage=50, radio=1):
    """
    Genera un círculo dividivo en dos segmentos circulares, según una proporción especificada.

    Parameters
    ----------
    pos : list/tuple, defaul=[0,0], opcional
        Coordenadas (x, y) de la posición del centro del círculo.
    left_percentage : float, default=50, opcional
        Porcentaje que abarca el área izquierda en relación al total del círculo.
    radio : float, default=1, opcional
        Radio del círculo.

    Returns
    -------
    list
        Lista de dos patches de matplotlib que en conjunto representan un círculo dividido.
    """
    percentage = left_percentage
    angle = 90 - (90 / 50) * percentage

    return [
        arc_patch((pos[0], pos[1]), radio, 270 - angle, 90 + angle),
        arc_patch((pos[0], pos[1]), radio, -90 - angle, 90 + angle),
    ]


def bubble_plot(
    ax,
    df,
    position_column,
    radius_column,
    label_column=None,
    palette="plasma",
    n_bins=10,
    num_steps=50,
    x_position_scaling=800,
    min_label_size=4,
    max_label_size=64,
    starting_y_range=None,
    margin=2,
    dual=False,
    dual_left_color="cornflowerblue",
    dual_right_color="hotpink",
    fontname=None,
    fontstyle=None,
    random_state=1990,
):
    """
    Crea un gráfico de burbujas a partir de los datos entregados.
    Esta visualización muestra los datos en forma de burbujas o círculos de diferentes tamaños en un eje.
    Cada burbuja representa una entidad o categoría y se posiciona en función
    del valor de la variable asociada al eje ` x` , mientras que el tamaño de la burbuja indica la magnitud de una segunda variable.
    Cuanto más grande sea la burbuja, mayor será el valor de la variable asociada.
    Una burbuja también tendrá un color que representa su ubicación en el eje.
    En este gráfico, la posición en el eje `y` de cada burbuja no es relevante.

    En el notebook notebooks/vis-course/06-python-texto-guaguas.ipynb se encuentran ejemplos de uso de esta función.


    Parameters
    -------------
    ax : matplotlib.axes
        El eje en el cual se dibujará el gráfico.
    df : DataFrame
        DataFrame que contiene los datos a visualizar.
    position_column : str
        Nombre de la columna que contiene el valor usado para posicionar una burbuja en el eje.
    radius_column : str
        Nombre de la columna que contiene el valor usado para determinar el tamaño de una burbuja.
    label_column : str, default=None, optional
        Nombre de la columna que contiene las etiquetas de la burbuja.
    palette : str, default="plasma", optional
        Paleta de colores utilizada para el mapa de colores.
    n_bins : int, default=10, optional
        Número de bins utilizados para dividir el mapa de colores
    num_steps : int, default=50, optional
        Número de pasos utilizados para la simulación de las burbujas.
    x_position_scaling : int, default=800, optional
        Factor de escala para ajustar la posición de las burbujas en el eje x.
    min_label_size : int, default=4, optional
        Tamaño mínimo de las etiquetas de las burbujas.
    max_label_size : int, default=64, optional
        Tamaño máximo de las etiquetas de las burbujas.
    starting_y_range : int, default=None, optional
        Rango inicial para la posición aleatoria en el eje y de las burbujas.
    margin : int, default=2, optional
        Margen utilizado para ajustar el tamaño de las burbujas.
    dual : bool, default=False, optional
        Indicador de si se debe generar una representación de burbujas duales. Si es `True`, en vez de utilizar un mapa de color para colorear
        una burbuja, esta estará coloreada por dos colores que representan cada extremo del eje. El área que abarca cada color en el círculo
        es determinada por la ubicación de la burbuja en el eje.
    dual_left_color : str, default="cornflowerblue", optional
        Color utilizado para la representación de burbujas duales en el lado izquierdo.
    dual_right_color : str, default="hotpink", optional
        Color utilizado para la representación de burbujas duales en el lado derecho.
    fontname : str, default=None, optional
        Nombre de la fuente utilizada para las etiquetas de las burbujas.
    fontstyle : str, default=None, optional
        Estilo de la fuente utilizada para las etiquetas de las burbujas.
    random_state : int, default=1990, optional
        Estado aleatorio utilizado para la generación de las burbujas.

    Returns
    -----------
    space : objeto Space de pymunk
        El espacio de simulación de pymunk utilizado para las burbujas.
    collection : objeto PatchCollection de matplotlib, opcional
        La colección de parches utilizada para dibujar las burbujas en el gráfico. Devuelto solo si dual es False.
    split_collection_l : objeto PatchCollection de matplotlib, opcional
        La colección de parches utilizada para dibujar las burbujas duales en el lado izquierdo del gráfico. Devuelto solo si dual es True.
    split_collection_r : objeto PatchCollection de matplotlib, opcional
        La colección de parches utilizada para dibujar las burbujas duales en el lado derecho del gráfico. Devuelto solo si dual es True.
    """
    np.random.seed(random_state)
    df = df.reset_index()
    space = pymunk.Space()
    space.gravity = (0, 0)

    radius = np.sqrt(df[radius_column].values) + margin

    if starting_y_range is None:
        starting_y_range = int(np.sqrt(df.shape[0]))

    for idx, row in df.iterrows():
        x = row[position_column] * x_position_scaling
        y = np.random.randint(-starting_y_range, starting_y_range)
        mass = 10
        r = radius[idx]

        moment = pymunk.moment_for_circle(mass, 0, r, (0, 0))

        body = pymunk.Body(mass, moment)

        body.position = x, y
        body.start_position = Vec2d(*body.position)

        shape = pymunk.Circle(body, r)
        shape.elasticity = 0.9999999

        space.add(body, shape)

    for i in range(num_steps):
        space.step(1)

    value_range = (min(radius), max(radius))

    def scale(value):
        result = (value - value_range[0] * 0.75) / (
            value_range[1] - value_range[0] * 0.75
        )
        return result

    ax.set_aspect("equal")

    cmap = ListedColormap(sns.color_palette(palette, n_colors=n_bins))

    collection = []
    arcs = []
    values = []
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    for body, (idx, row) in zip(space.bodies, df.iterrows()):
        circle = next(iter(body.shapes))

        body_x = body.position.x
        body_min_x = body_x - circle.radius
        body_max_x = body_x + circle.radius

        body_y = body.position.y
        body_min_y = body_y - circle.radius
        body_max_y = body_y + circle.radius

        if body_min_y < min_y:
            min_y = body_min_y

        if body_min_x < min_x:
            min_x = body_min_x

        if body_max_y > max_y:
            max_y = body_max_y

        if body_max_x > max_x:
            max_x = body_max_x

        if dual == True:
            arcs.append(
                build_dual_bubble(
                    pos=list(body.position),
                    radio=circle.radius,
                    # left_percentage=100-100*(body.position.x-min_x)/(max_x-min_x)
                    # left_percentage=row.M*100
                    left_percentage=100 - 100 * (row[position_column] + 1) / 2,
                )
            )
        else:
            c = Circle(np.array(body.position), circle.radius - margin)
            collection.append(c)
            values.append(row[position_column])
        if label_column is not None:
            label_size = int(scale(radius[idx]) * max_label_size)
            if label_size < min_label_size:
                continue
            ax.annotate(
                row[label_column],
                np.array(body.position),
                ha="center",
                va="center",
                fontsize=label_size,
                fontname=fontname,
                fontstyle=fontstyle,
            )

    if dual == True:
        split_collection_l = PatchCollection(
            (a[0] for a in arcs), facecolors=dual_left_color, edgecolor="none"
        )
        ax.add_collection(split_collection_l)
        split_collection_r = PatchCollection(
            (a[1] for a in arcs), facecolors=dual_right_color, edgecolor="none"
        )
        ax.add_collection(split_collection_r)
        collection = None
    else:
        collection = PatchCollection(
            collection, color="pink", edgecolor="none", cmap=cmap
        )
        collection.set_array(np.array(values))
        ax.add_collection(collection)
        split_collection_l = None
        split_collection_r = None

    ax.set_aspect("equal")
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])

    return space, collection, split_collection_l, split_collection_r
