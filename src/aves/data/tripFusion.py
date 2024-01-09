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
