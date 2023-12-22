
def crear_datos_regresion(trip_counts, zone_metro_nearby, treatment_zones):
    """
    Combina y procesa datos para el análisis de regresión.

    Parameters:
    - trip_counts (pandas.DataFrame): DataFrame con los conteos de viajes.
    - zone_metro_nearby (pandas.DataFrame): DataFrame con información de zonas cercanas al metro.
    - treatment_zones (pandas.DataFrame): DataFrame con los conteos de viajes de scooters por zona.

    Returns:
    - pandas.DataFrame: DataFrame resultante con los datos procesados para el análisis de regresión.
    """
    regression_data = (
        trip_counts.merge(zone_metro_nearby, left_on="diseno777subida", right_index=True, how="left")
        .merge(treatment_zones, on=["year", "dayofweek", "hour", "diseno777subida"], how="left")
        .pipe(lambda x: x[x["diseno777subida"] != "-"])
        .assign(scooters=lambda x: x["scooter_trips"] > 0)
        .drop_duplicates()
    )

    return regression_data

"""
En el caso de estudio se usa:
regression_data = crear_datos_regresion(trip_counts, zone_metro_nearby, treatment_zones)
"""