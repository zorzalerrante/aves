import matplotlib.pyplot as plt
import seaborn as sns

from aves.features.utils import normalize_rows


def barchart(
    ax,
    df,
    categories=None,
    palette="plasma",
    stacked=False,
    normalize=False,
    sort_items=False,
    sort_categories=False,
    fill_na_value=None,
    bar_width=0.9,
    legend=True,
    legend_args=None,
    return_df=False,
    **kwargs
):
    """ 
    Crea un gráfico de barras a partir de los datos del dataframe. Un gráfico de barras muestra comparaciones entre categorías
    discretas. Uno de los ejes del gráfico muestra las categorías específicas que se están comparando,
    y el otro eje corresponde a un valor numérico medido. Cada rectánguo del gráfico representa una categoría, y su altura
    es proporcional al valor que toma esa categoría.

    En el notebook `notebooks/vis-course/02-python-tablas.ipynb` se pueden encontrar ejemplos de uso
    de esta función.

    Parameters
    -----------------
    ax : matplotlib.axes
        El eje en el cual se dibujará el gráfico.
    df: Pandas.dataframe
        Datos a visualizar. El índice del dataframe  usar como eje categórico.
    categories: str o list o posición, default=None, opcional
        Qué columnas del dataframe graficar. Si no se especifica, se usarán todas las comunas de tipo numérico.
    palette : str o list, default="Plasma", opcional
        La paleta de colores a utilizar en el gráfico.
    stacked : bool, default=False, opcional
        Indica si las barras se deben apilar una encima de la otra. Si se elige esta opción, cada barra se
        dividirá en segmentos que representan una categoría o variable diferente
    normalize : bool, default=False, opcional
        Indica si se deben normalizar los valores en el DataFrame.
    sort_items : bool, default=False, opcional
        Indica si se deben ordenar las barras por el primer valor de columna.
    sort_categories : bool, default=False, opcional
        Indica si se deben ordenar las categorías por el valor medio de cada columna.
    fill_na_value : objeto, default=None, opcional
        Valor utilizado para rellenar los valores faltantes en el DataFrame. Por defecto es None y los valores faltantes no se rellenan.
    bar_width : float, default=0.9, opcional
        El ancho de las barras en el gráfico.
    legend : bool, default=True, opcional
        Indica si se muestra la leyenda en el gráfico.
    legend_args : dict, default=None, opcional
        Argumentos adicionales para personalizar la leyenda.
    return_df : bool, default=None, opcional
        Indica si se debe devolver el DataFrame utilizado en el gráfico. 
    **kwargs : dict
        Argumentos adicionales para personalizar el gráfico que se pasan a la función `plot.bar()` de pandas.
        Una lista completa de todas las posibles especificaciones se encuentra en la documentación de `Matplotlib <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html>`__
    

    Returns
    -------
    pd.DataFrame
        En caso de invocar la función con el argumento return_df=True, retorna "df" procesado según los otros parámetros.
    """
    sns.set_palette(palette, n_colors=len(df.columns))

    if categories is not None:
        df = df[categories]

    if fill_na_value is not None:
        df = df.fillna(fill_na_value)

    if normalize:
        df = df.pipe(normalize_rows)

    if sort_categories:
        sort_values = df.mean(axis=0).sort_values(ascending=False)
        df = df[sort_values.index].copy()

    if sort_items:
        df = df.sort_values(df.columns[0])

    df.plot.bar(
        ax=ax,
        stacked=stacked,
        width=bar_width,
        edgecolor="none",
        legend=legend,
        **kwargs
    )

    if legend:
        if legend_args is None:
            legend_args = dict(
                bbox_to_anchor=(1.0, 0.5), loc="center left", frameon=False
            )
        handles, labels = map(reversed, ax.get_legend_handles_labels())
        ax.legend(handles, labels, **legend_args)

    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    sns.despine(ax=ax, left=True)

    if normalize:
        ax.set_ylim([0, 1])

    if return_df:
        return df
