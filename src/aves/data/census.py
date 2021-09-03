from pathlib import Path

import geopandas as gpd

_DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data"
_CENSUS_MAPS = _DATA_PATH / "external" / "censo_2017_R13"


def read_census_map(level, path=None):
    if path is None:
        DATA_PATH = _CENSUS_MAPS
    else:
        DATA_PATH = Path(path)

    return gpd.read_file(DATA_PATH / "{}_C17.shp".format(level.upper()))
