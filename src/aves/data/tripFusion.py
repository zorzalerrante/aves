
def calcular_trip_counts(dataframe):
    """
    Calcula el conteo de viajes agrupados por año, día de la semana, hora, diseño de la subida y comuna de la subida.

    Parameters:
    - dataframe (dask.dataframe.DataFrame): El DataFrame Dask que contiene los datos de los viajes.

    Returns:
    - dask.dataframe.DataFrame: DataFrame resultante con el conteo de viajes, agregado y calculado.
    """
    trip_counts = (
        dataframe.assign(hour=lambda x: x["tiemposubida"].dt.hour)
        .groupby(['year', 'dayofweek', "hour", "diseno777subida", "comunasubida"])["factorexpansion"]
        .sum()
        .rename("n_viajes")
        .reset_index()
        .assign(tiempo=0)
        .compute()
    )
    return trip_counts

"""
En el caso de estudio usamos:
trip_counts = calcular_trip_counts(ViajesTotales)
"""

