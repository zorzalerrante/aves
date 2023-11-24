from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np

_DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data"
_CENSUS_MAPS = _DATA_PATH / "external" / "censo_2017/geometria"
_CENSUS_PATH = _DATA_PATH / "external" / "censo_2017"


def read_census_map(level, path=None):
    if path is None:
        DATA_PATH = _CENSUS_MAPS
    else:
        DATA_PATH = Path(path)

    return gpd.read_file(DATA_PATH / "{}_C17.shp".format(level.upper()))


def read_comuna(path=None):
    """
    Carga la geometría de las comunas de la Región Metropolitana definidas en el censo
    2017, a partir del archivo "COMUNA_C17.shp".

    Parameters
    ----------
    path: string, default=None
        Ubicación del archivo "COMUNA_C17.shp" que contiene las geometrías.

    Returns
    -------
    gpd.GeoDataFrame
        Dataframe con la geometría de cada comuna de la Región Metropolitana,
        su nombre, la provincia y región a la que pertenece y el área que abarca.
    
    """
    if path is None:
        DATA_PATH = _CENSUS_MAPS
    else:
        DATA_PATH = Path(path)

    return gpd.read_file(DATA_PATH / "COMUNA_C17.shp")


def read_distrito(path=None):
    """
    Carga la geometría de los distritos de la Región Metropolitana definidos en el censo
    2017, a partir del archivo "DISTRITO_C17.shp".

    Parameters
    ----------
    path: string, default=None
        Ubicación del archivo "DISTRITO_C17.shp" que contiene las geometrías.

    Returns
    -------
    gpd.GeoDataFrame
        Dataframe con la geometría de cada distrito de la Región Metropolitana,
        su nombre, la comuna, provincia y región a la que pertenece, el tipo de distrito
        y el área que abarca.
    
    """
    if path is None:
        DATA_PATH = _CENSUS_MAPS
    else:
        DATA_PATH = Path(path)

    return gpd.read_file(DATA_PATH / "DISTRITO_C17.shp")


def read_region(path=None):
    """
    Carga la geometría de la Región Metropolitana definida en el Censo 2017 a partir del archivo "REGION_C17.shp".

    Parameters
    ----------
    path: string, default=None
        Ubicación del archivo "REGION_C17.shp" que contiene las geometrías.

    Returns
    -------
    gpd.GeoDataFrame
        Dataframe con la geometría de  la Región Metropolitana.
    
    """
    if path is None:
        DATA_PATH = _CENSUS_MAPS
    else:
        DATA_PATH = Path(path)

    return gpd.read_file(DATA_PATH / "REGION_C17.shp")


def read_localidad(path=None):
    """
    Carga la geometría de las localidades de la Región Metropolitana definidas en el censo
    2017, a partir del archivo "LOCALIDAD_C17.shp".

    Parameters
    ----------
    path: string, default=None
        Ubicación del archivo "LOCALIDAD_C17.shp" que contiene las geometrías.

    Returns
    -------
    gpd.GeoDataFrame
        Dataframe con la geometría de cada localidad de la Región Metropolitana,
        su nombre, el distrito, comuna, provincia y región a la que pertenece,
        y el área que abarca.
    
    """
    if path is None:
        DATA_PATH = _CENSUS_MAPS
    else:
        DATA_PATH = Path(path)

    return gpd.read_file(DATA_PATH / "LOCALIDAD_C17.shp")


def read_provincia(path=None):
    """
    Carga la geometría de las provincias de la Región Metropolitana definidas en el censo
    2017, a partir del archivo "PROVINCIA_C17.shp".

    Parameters
    ----------
    path: string, default=None
        Ubicación del archivo "PROVINCIA_C17.shp" que contiene las geometrías.

    Returns
    -------
    gpd.GeoDataFrame
        Dataframe con la geometría de cada provincia de la Región Metropolitana,
        su nombre, la región a la que pertenece, y el área que abarca.
    
    """
    if path is None:
        DATA_PATH = _CENSUS_MAPS
    else:
        DATA_PATH = Path(path)

    return gpd.read_file(DATA_PATH / "PROVINCIA_C17.shp")


def read_zona(path=None):
    """
    Carga la geometría de las zonas de la Región Metropolitana definidas en el censo
    2017, a partir del archivo "ZONA_C17.shp".

    Parameters
    ----------
    path: string, default=None
        Ubicación del archivo "ZONA_C17.shp" que contiene las geometrías.

    Returns
    -------
    gpd.GeoDataFrame
        Dataframe con la geometría de cada zona de la Región Metropolitana,
        su nombre, la localidad, comuna, provincia y región a la que pertenece,
        y el área que abarca.
    
    """
    if path is None:
        DATA_PATH = _CENSUS_MAPS
    else:
        DATA_PATH = Path(path)

    return gpd.read_file(DATA_PATH / "ZONA_C17.shp")


def read_entidad(path=None):
    #TODO: averiguar qué es esto, hay categorias como Parcela de Agrado ,Parcela-Hijuela  , Caserío y cantidad de viviendas/habitantes
    """

    Parameters
    ----------
    path: string, default=None
        Ubicación del archivo "ENTIDAD_IND_C17.shp" que contiene las geometrías.

    Returns
    -------
    gpd.GeoDataFrame
        Dataframe 
    
    """
    if path is None:
        DATA_PATH = _CENSUS_MAPS
    else:
        DATA_PATH = Path(path)

    return gpd.read_file(DATA_PATH / "ENTIDAD_IND_C17.shp")


def read_limite(path=None):
    """
    Carga la geometría de los limites _____ Región Metropolitana definidos en el censo
    2017, a partir del archivo "LIM_DPA_CENSAL_C17.shp".

    Parameters
    ----------
    path: string, default=None
        Ubicación del archivo "LIM_DPA_CENSAL_C17.shp" que contiene las geometrías.

    Returns
    -------
    gpd.GeoDataFrame
        Dataframe con la geometría de los límites, el tipo de límite y una descripción.
    
    """
    if path is None:
        DATA_PATH = _CENSUS_MAPS
    else:
        DATA_PATH = Path(path)

    return gpd.read_file(DATA_PATH / "LIM_DPA_CENSAL_C17.shp")


def read_limite_urbano_censal(path=None):
    """
    Carga la geometría de los limites _____ Región Metropolitana definidos en el censo
    2017, a partir del archivo "LIMITE_URBANO_CENSAL_C17.shp".

    Parameters
    ----------
    path: string, default=None
        Ubicación del archivo "LIMITE_URBANO_CENSAL_C17.shp" que contiene las geometrías.

    Returns
    -------
    gpd.GeoDataFrame
        Dataframe con la geometría de los límites, el tipo de límite y .
    
    """
    if path is None:
        DATA_PATH = _CENSUS_MAPS
    else:
        DATA_PATH = Path(path)

    return gpd.read_file(DATA_PATH / "LIMITE_URBANO_CENSAL_C17.shp")


def read_manzana_aldea(path=None):
    """
    Carga la geometría de las  de la Región Metropolitana definidas en el censo
    2017, a partir del archivo "MANZANA_ALDEA_C17.shp".

    Parameters
    ----------
    path: string, default=None
        Ubicación del archivo "MANZANA_ALDEA_C17.shp" que contiene las geometrías.

    Returns
    -------
    gpd.GeoDataFrame
        Dataframe con la geometría.
    
    """
    if path is None:
        DATA_PATH = _CENSUS_MAPS
    else:
        DATA_PATH = Path(path)

    return gpd.read_file(DATA_PATH / "MANZANA_ALDEA_C17.shp")


def get_urban_zones(zones, remove_artifacts=True, area_threshold=100000):
    """
    Filtra una zona geográfica, retornando aquellas zonas que están dentro del límite urbano censal según
    el Censo 2017. De momento sólo funciona para la Región Metropolitana.

    Parameters
    ----------
    zones: geopandas.dataframe
        Zona geográfica a filtrar.
    remove_artifacts: boolean, default=True
        Indica si se deben eliminar las zonas cuya área es menor a `area_threshold` del resultado.
    area_threshold: float, default=100000
        Área mínima que debe tener una zona para ser parte del dataframe resultante si es que
        `remove_artifacts` es True. El valor por defecto se expresa en metros cuadrados.

    Returns
    -------
    geopandas.DataFrame
        Zonas de dataframe original que corresponden a una zona urbana.
    """
    urban_zones = read_limite_urbano_censal()
    clipped_zones = gpd.overlay(zones, urban_zones.to_crs(zones.crs), how='intersection')
    if remove_artifacts:
        clipped_zones['area_m2'] = clipped_zones.area
        clipped_zones = clipped_zones[clipped_zones['area_m2'] > area_threshold]
    return clipped_zones


def get_urban_municipalities():
    comunas = read_comuna()
    zones = read_limite_urbano_censal().to_crs(comunas.crs)
    comunas_urbanas = (
        comunas[comunas["COMUNA"].isin(zones["COMUNA"].unique())]
        .drop("NOM_COMUNA", axis=1)
        .copy()
    )

    comunas_urbanas["NombreComuna"] = comunas_urbanas["COMUNA"].map(
        dict(zip(zones["COMUNA"], zones["NOM_COMUNA"].str.upper()))
    )

   # bounding_box = zones.total_bounds
    #comunas_urbanas = clip_area_geodataframe(comunas_urbanas, zones.total_bounds, buffer=1000)
    comunas_urbanas['NombreComuna'] = comunas_urbanas['NombreComuna'].replace({'Á': 'A', 'Ú': 'U', 'Ó': 'O', 'Í': 'I', 'É': 'E'}, regex=True)
    return comunas_urbanas


def municipalities_in_box(bbox):
    return None


def decode_column(
    df,
    fname,
    col_name,
    index_col="Id",
    value_col=None,
    sep=";",
    encoding="utf-8",
    index_dtype=np.float64,
):
    """
    Decodifica los valores de una columna, reemplazando identificadores por su correspondiente valor según la tabla de códigos.

    Parameters
    ------------
    df : pandas.dataframe
        Dataframe del que se leerá una columna.
    fname: string
        Nombre del archivo que contiene los valores a decodificar.
    col_name: string
        Nombre de la columna que queremos decodificar.
    index_col: string, default="Id"
        Nombre de la columna en el archivo ` fname `  que tiene los índices que codifican ` col_name ` .
    value_col: string, default=None
        Nombre de la columna en el archivo ` fname `  que tiene los valores decodificados.
    sep: string, default=";"
        Caracter que separa los valores en ` fname ` .
    encoding: string, default="utf-8"
        Identificación del character set que utiliza el archivo. Usualmente es utf-8, si no funciona,
          se puede probar con iso-8859-1. 
    index_dtype: dtype, default=np.float64

    Returns
    -------
    pd.DataFrame
        Dataframe decodificado en la columna señalada.
    
    """
    if value_col is None:
        value_col = "value"

    values_df = pd.read_csv(
        fname,
        sep=sep,
        index_col=index_col,
        names=[index_col, value_col],
        header=0,
        dtype={index_col: index_dtype},
        encoding=encoding,
    )

    src_df = df.loc[:, (col_name,)]

    return src_df.join(values_df, on=col_name)[value_col]


def process_hogares(
        path=None
):
    """
    Procesa el contenido del archivo "Hogares.csv" y lo almacena en un
    archivo de formato Parquet.
    
    """
    if path is None:
        DATA_PATH = _CENSUS_PATH
    else:
        DATA_PATH = Path(path)
    df = pd.read_csv(
        DATA_PATH / "Hogares.csv", sep=";", decimal=",", encoding="utf-8")
    df['TIPO_OPERATIVO']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/TIPO_OPERATIVO.csv", index_col="Id", col_name="TIPO_OPERATIVO", sep=',')
    df['TIPO_HOGAR']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/TIPO_HOGAR.csv", index_col="Id", col_name="TIPO_HOGAR", sep=',')
    traduccion = pd.read_csv(DATA_PATH/"Tablas_parametros/traduccion_nuble.csv")
    cols_to_replace = ["COMUNA", "REGION", "PROVINCIA"]
    for col in cols_to_replace:
        df[col] = df['ID_ZONA_LOC'].map(traduccion.set_index('ID_ZONA_LOC')[col])
    df['NOM_COMUNA']=decode_column(df, fname=DATA_PATH/"Tablas_parametros/COMUNA.csv", index_col="COMUNA", col_name="COMUNA", sep=';')

    df.to_parquet(DATA_PATH/"Hogares.parquet", index=False)


def read_hogares(
    path=None,
    regiones=None,
    comunas=None,
    columnas=None
):
    """
    Carga el contenido del archivo "Hogares.parquet", que contiene las respuestas sobre hogares, a un dataframe.
    Los hogares corresponden a la manera de organización de las personas dentro de las viviendas
    particulares y constituyen por derecho propio una unidad de empadronamiento en sí mismos.

    Parameters
    ----------
    path: string, default=None
        Ubicación del archivo parquet con la data del censo 2017.
    regiones: list, default=None
        Las regiones a incluir. Si no se especifica, se cargarán todas
    comunas: list, default=None
        Las comunas a incluir. Si no se especifica, se cargarán todas
    columnas: list, default=None
        Si no es None, solo se cargarán las columnas especificadas.

    Returns
    -------
    pd.DataFrame
        Dataframe con la información sobre hogares, con las columnas decodificadas. 
    """
    if path is None:
        DATA_PATH = _CENSUS_PATH
    else:
        DATA_PATH = Path(path)
    try:
        df = pd.read_parquet(
        DATA_PATH / "Hogares.parquet", columns=columnas)
    except FileNotFoundError:
        process_hogares(path)
    if regiones:
        if type(regiones) is not list:
            regiones = [regiones]
            df = df[df.REGION.isin(regiones)]
    elif comunas:
        if type(comunas) is not list:
                    comunas = [comunas]
                    df = df[df.REGION.isin(comunas)]