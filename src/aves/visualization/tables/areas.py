import numpy as np
from aves.visualization.collections import LabelCollection
import seaborn as sns


def stacked_areas(ax, df, baseline="zero", color_dict=None, **kwargs):
    """
    Dibuja la visualización de la distribución cumulativa de distintas categorías a lo largo de una variable continua y obtiene
    los parámetros de la figura necesarios para generar un gráfico con la función `streamgraph`_.

    Parameters
    ----------
    ax : matplotlib.axes
        El eje en el cual se dibujará el gráfico.
    df : DataFrame
        Un DataFrame que contiene los datos para generar el gráfico.
    baseline : str, default="zero", opcional
        El método utilizado para calcular la línea base de las áreas apiladas.
    color_dict : dict, default=None, opcional
        Un diccionario que mapea los nombres de las categorías a los colores a utilizar para rellenar las áreas corresppondientes.
    **kwargs
        Argumentos adicionales que permiten personalizar el gráfico.
        Una lista completa de las opciones disponibles se encuentra en la documentación de `Matplotlib <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.fill_between.html>`__.

    Returns
    -------
    x : ndarray
        Arreglo que contiene los valores en el eje x del gráfico.
    first_line : ndarray
        Arreglo que contiene los valores de la primera línea base de las áreas apiladas.
    stack : ndarray
        Arreglo que contiene los valores de las áreas apiladas.
    """
    stack = np.cumsum(df.T.values, axis=0)
    x = df.index.values
    y = df.T.values
    m = y.shape[0]

    if baseline == "zero":
        first_line = np.zeros(x.shape)
    else:
        first_line = (y * (m - 0.5 - np.arange(m)[:, None])).sum(0)
        first_line /= -m
        first_line -= np.min(first_line)
        stack += first_line

    color = color_dict[df.columns[0]] if color_dict is not None else None
    ax.fill_between(x, first_line, stack[0, :], facecolor=color, **kwargs)

    for i in range(len(y) - 1):
        color = color_dict[df.columns[i + 1]] if color_dict is not None else None
        ax.fill_between(x, stack[i, :], stack[i + 1, :], facecolor=color, **kwargs)

    return x, first_line, stack


def streamgraph(
    ax,
    df,
    baseline="wiggle",
    labels=True,
    label_threshold=0,
    label_args=None,
    fig=None,
    facecolor=None,
    edgecolor=None,
    linewidth=None,
    palette=None,
    area_colors=None,
    area_args=None,
    avoid_label_collisions=False,
    outline_labels=True,
    label_collision_args=None,
):
    """
    Genera un gráfico **streamgraph** a partir de los datos de un dataframe.
    Este gráfico muestra el cambio de composición o distribución de distitnas categorías a lo largo del tiempo.
    Cada categoría es representada por una franja de un color que fluye por el eje horizontal, que representa una variable continua
    como por ejemplo el paso del tiempo. La altura de la franja en un punto representa la proporción relativa de esa categoría en ese momento.
    Las franjas están apiladas una encima de la otra, por lo que la altura total de las franjas en un punto indica el cumulativo de todas las categorias.

    En el notebook notebooks/vis-course/06-python-texto-guaguas.ipynb se encuentran ejemplos de uso de esta función.

    Parameters
    ----------
    ax : matplotlib.axes
        El eje en el cual se dibujará el gráfico.
    df : DataFrame
        DataFrame que contiene los datos a visualizar. Cada columna es una categoría.
    baseline : str, default="wiggle", opcional
        El método utilizado para calcular la línea base del streamgraph.
    labels : bool, default=True, opcional
        Indica si se deben mostrar etiquetas en el gráfico.
    label_threshold : int, default=0, opcional
        Umbral para posicionar las etiquetas en el gráfico.
    label_args : dict, default=None, opcional
        Argumentos adicionales para personalizar las etiquetas.
        Una lista completa de las opciones disponibles se encuentra en la documentación de `Matplotlib <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html>`__.
    fig : Figure, default=None, opcional
        La figura en la cual se genera el gráfico. Se utiliza para el manejo de colisiones de etiquetas.
    facecolor: string, default=None, opcional
        Un color para pintar todas las áreas. Su uso anula el de palette y area_colors.
    edgecolor: string, default=None, opcional
        Un color para pintar los bordes de las áreas.
    linewidth: float, default=None, opcional
        El grosor de la línea borde de cada área.
    palette: string, default=None, opcional
        Un nombre de paleta de colores para colorear cada área. Solo se utiliza si area_colors es None.
    area_colors : dict, default=None, opcional
        Un diccionario que mapea los nombres de las categorías a los colores a utilizar para rellenar las áreas corresppondientes.
    area_args : dict, default=None, opcional
        Argumentos adicionales que permiten personalizar el gráfico.
        Una lista completa de las opciones disponibles se encuentra en la documentación de `Matplotlib <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.fill_between.html>`__.
    avoid_label_collisions : bool, default=False, opcional
        Indica si se deben evitar colisiones de etiquetas en el gráfico.
    outline_labels : bool, default=True, opcional
        Indica si se deben resaltar las etiquetas mediante un contorno.
    label_collision_args : dict, default=None, opcional
        Argumentos adicionales para manejar las colisiones de etiquetas.
        Una lista completa de las opciones disponibles se encuentra en la documentación de `AdjustText <https://adjusttext.readthedocs.io/en/latest/>`__.
    """
    if label_args is None:
        label_args = {}

    if area_args is None:
        area_args = {}

    if label_collision_args is None:
        label_collision_args = dict(
            iter_lim=25, arrowprops=dict(arrowstyle="-", color="k", lw=0.5)
        )

    df = df.fillna(0).astype(float)

    if area_colors is None and palette is not None:
        area_colors = dict(
            zip(df.columns.values, sns.color_palette(palette, n_colors=len(df)))
        )

    stream_x, stream_first_line, stream_stack = stacked_areas(
        ax, df, color_dict=area_colors, baseline=baseline, **area_args
    )

    if labels:
        x_to_idx = dict(zip(df.index, range(len(df))))
        max_x = df.idxmax().map(x_to_idx)

        label_collection = LabelCollection()

        max_idx = max_x.values[0]
        y_value = stream_stack[0, max_idx] - stream_first_line[max_idx]

        if y_value >= label_threshold:
            label_collection.add_text(
                df.columns[0],
                stream_x[max_idx],
                stream_first_line[max_idx] * 0.5 + stream_stack[0, max_idx] * 0.5,
            )

        for i in range(1, len(df.columns)):
            max_idx = max_x.values[i]
            y_value = stream_stack[i, max_idx] - stream_stack[i - 1, max_idx]

            if y_value < label_threshold:
                continue

            label_collection.add_text(
                df.columns[i],
                stream_x[max_idx],
                stream_stack[i, max_idx] * 0.5 + stream_stack[i - 1, max_idx] * 0.5,
            )

        label_collection.render(
            ax,
            fig=fig,
            color=label_args.get("color", "white"),
            fontweight=label_args.get("fontweight", "bold"),
            fontsize=label_args.get("fontsize", "medium"),
            outline=outline_labels,
            avoid_collisions=avoid_label_collisions,
            adjustment_args=label_collision_args,
        )
