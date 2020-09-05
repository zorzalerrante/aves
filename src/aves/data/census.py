import pandas as pd
import geopandas as gpd
from pathlib import Path


_DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / 'data'
_CENSUS_MAPS = _DATA_PATH / 'external' / 'censo_2017_R13'

def read_census_map(level, path=None):
    if path is None:
        DATA_PATH = _CENSUS_MAPS
    else:
        DATA_PATH = path
        
    return gpd.read_file(_CENSUS_MAPS / '{}_C17.shp'.format(level.upper()))