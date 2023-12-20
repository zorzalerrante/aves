
def year_day_hour(df):

    semana2018 = semana2018.assign(
        hour=lambda x: x["fecha"].dt.hour,
        dayofweek=lambda x: x["fecha"].dt.dayofweek,
        year = lambda x: x["fecha"].dt.year,
        tiempo=1
    )

"""
En nuestro caso hacemos
semana2018 = semana2018.reset_index(drop=True)
semana2019 = semana2019.reset_index(drop=True)

ViajesTotales = dd.concat([semana2018, semana2019], axis=0)

year_day_hour(ViajesTotales)
"""

