from datetime import datetime
import pandas as pd
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import numpy as np
import seaborn as sns
import dask.dataframe as dd
from datetime import datetime

DATA_PATH = "/ruta/a/la/carpeta"

from pathlib import Path

AVES_ROOT = Path("/ruta a la carpeta")
DATA_PATH = Path("/ruta a la carpeta")
SCOOTERS_PATH = Path("/ruta a la carpeta")
ADATRAP_PATH = Path("/ruta a la carpeta")

#Recibe la fecha desde la cual se quiere leer los viajes (start_date) hasta (end_date)
def read_date(date, data_path):
    return dd.read_parquet(data_path/ f"{date}")

def read_concat_date(start_date, end_date, data_path, output_name):
    """
    Recibe la fecha desde la cual se quiere leer los viajes (start_date) hasta (end_date),
    los concatena y los guarda en un archivo parquet para luego ser leído directamente.

    Parameters
    ------------
    start_date : string
        Fecha desde la cual se comenzarán a leer los días.
    end_date: string
        Fecha hasta la cual se comenzarán a leer los días.
    data_path: Path
        Ruta de los archivos a leer y donde se guardara el nuevo arhivo concatenado.
    output_name: string
        Nombre del archivo resultante con la concatenación.

    Returns
    -------
    df_concatenado
        Dataframe con todas las fechas con el atributo que lo especifica
    
    """
    # Genera las fechas entre start_date y end_date
    date_range = [datetime.strftime(d, "%Y-%m-%d") for d in pd.date_range(start=start_date, end=end_date)]

    # Lee los dataframs y agrega una columna de fecha
    dataframes = []
    for fecha in date_range:
        df = dd.read_parquet(data_path / fecha)
        df['fecha'] = datetime.strptime(fecha, "%Y-%m-%d")
        dataframes.append(df)

    # Concatena los dataframes
    df_concatenado = dd.concat(dataframes, axis=0)

    # Escribe el dataframe en un archivo Parquet
    df_concatenado.to_parquet(data_path / f"{output_name}.parquet")

"""
para la semana 2018 y 2019 correspondería:
read_concat_date('2018-05-14', '2018-05-20', DATA_PATH, "semana2018")
read_concat_date('2019-05-13', '2019-05-19', DATA_PATH, "semana2019")
"""

from pyproj import CRS

def read_zones(data_path):
    # La función lee los archivos pues uno contiene las geometrías y el otro no
    zones_dbf = gpd.read_file(Path(data_path) / 'ZONAS777_V07_04_2014.DBF')
    zones_shp = gpd.read_file(Path(data_path) / 'Zonas777_V07_04_2014.shp')

    # Asegura que la columna 'ZONA777' es de tipo str
    zones_dbf = zones_dbf.assign(ZONA777=lambda x: x["ZONA777"].astype(str))

    # Copia la columna geometry de zones_shp a zones_dbf
    zones_dbf['geometry'] = zones_shp['geometry']

    # Asegura que ambos tengan el mismo CRS
    zones_dbf.crs = zones_shp.crs
    crs_target = CRS.from_epsg(4326)
    zones_dbf = zones_dbf.set_crs(epsg=4326, inplace=True)
    zones_dbf = zones_dbf.to_crs(crs_target)

    return zones_dbf

"""
Para nuestro objetivo
zones = read_zones(file_path)
"""

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