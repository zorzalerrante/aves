
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

import statsmodels.formula.api as sm

def Analisis_ols(dataframe):
    """
    Realiza un análisis de regresión OLS sobre el DataFrame proporcionado.

    Parameters:
    - dataframe (pandas.DataFrame): DataFrame con los datos para el análisis de regresión.

    Returns:
    - statsmodels.iolib.summary.Summary: Resumen del análisis de regresión OLS.
    """
    model_formula = "np.log(n_viajes + 1) ~ 1 + C(hour) + year + dayofweek + hour + scooters + L1 + L2 + L3 + L4 + L4A + L5 + L6 + LCentralSur + Metrotren"
    regression_model = sm.ols(model_formula, data=dataframe).fit()
    summary = regression_model.summary()

    return summary

"""
En el caso de estudio se usa:
Analisis_ols(regression_data)
"""