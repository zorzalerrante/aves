import matplotlib.pyplot as plt
import seaborn as sns

from aves.visualization.collections.labels import LabelCollection


def scatterplot(
    ax,
    df,
    x,
    y,
    hue=None,
    annotate=False,
    avoid_collisions=False,
    scatter_args={},
    label_args={},
    na_value=0,
    drop_na=False,
    label_collision_args={"lim": 5},
    label_filter_func=None,
    legend_args=None,
):
    """
    Genera un scatter plot a partir de los datos entregados.
    Este gráfico visualiza los valores de dos variables para un set de datos como puntos en un plano bidimensional.
    

    Parameters
    ----------
    ax : matplotlib.axes
    El eje en el cual se dibujará el gráfico.
    df: Pandas.dataframe
    Datos a visualizar.
    x: str
        El nombre de la columna que contiene los valores del eje x.
    y: str
        El nombre de la columna que contiene los valores del eje y.
    hue: str, default=None
        El nombre de la columna que contiene los valores para agrupar los puntos.
        Si se proporciona, los puntos del gráfico se colorearán según los grupos. 
        La columna puede contener datos numéricos o categóricos.
    annotate: bool, default=False
        Indica si se deben agregar etiquetas a los puntos del gráfico.
    avoid_collisions: bool, default=False
        Indica si se deben evitar las colisiones entre las etiquetas al agregarlas al gráfico.
    scatter_args: dict, default={}
        Argumentos adicionales que permiten personalizar el gráfico.
        Una lista completa de las opciones disponibles se encuentra en la documentación de `SeaBorn <https://seaborn.pydata.org/generated/seaborn.scatterplot.html>`__.
    label_args: dict, default={}
    Argumentos adicionales para personalizar las etiquetas.
        Una lista completa de las opciones disponibles se encuentra en la documentación de `Matplotlib <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html>`__.
    na_value: int or float, default=0
        Valor a utilizar para reemplazar los valores faltantes en el DataFrame, si se opta por no eliminar las filas con valores faltantes.
    drop_na: bool, default=False
        Indica si se deben eliminar las filas con valores faltantes del DataFrame.
    label_collision_args: dict, default={"lim": 5}
        Argumentos adicionales para manejar las colisiones de etiquetas.
        Una lista completa de las opciones disponibles se encuentra en la documentación de `AdjustText <https://adjusttext.readthedocs.io/en/latest/>`__.
    label_filter_func: function, default=None
        Una función que filtra el DataFrame antes de agregar las etiquetas. Si se proporciona, solo se agregarán etiquetas para las filas que cumplan con los criterios de esta función.

    Returns
    -------------
    None
"""
    if not drop_na:
        df = df.fillna(na_value)
    else:
        df = df.dropna()

    g = sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, **scatter_args)
    if legend_args is not None:
        g.legend(**legend_args)
    ax.ticklabel_format(useOffset=False, style="plain")
    sns.despine(ax=ax)

    if annotate:
        collection = LabelCollection()

        if label_filter_func is not None:
            label_df = df.pipe(label_filter_func)
        else:
            label_df = df

        for index, row in label_df.iterrows():
            collection.add_text(index, row[x], row[y])

        collection.render(
            ax,
            avoid_collisions=avoid_collisions,
            adjustment_args=label_collision_args,
            **label_args
        )
