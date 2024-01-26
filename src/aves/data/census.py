from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import dask.dataframe as dd

_DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data"
_CENSUS_MAPS = _DATA_PATH / "external" / "censo_2017/geometria"
_CENSUS_PATH = _DATA_PATH / "external" / "censo_2017"


def read_census_map(level, path=None):
    if path is None:
        DATA_PATH = _CENSUS_MAPS
    else:
        DATA_PATH = Path(path)

    return gpd.read_file(DATA_PATH / "{}_C17.shp".format(level.upper()))


def read_comuna(region, path=None):
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

    return gpd.read_file(DATA_PATH / f"R{region:0>2}" / "COMUNA_C17.shp")


def read_distrito(region, path=None):
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

    return gpd.read_file(DATA_PATH / f"R{region:0>2}" / "DISTRITO_C17.shp")


def read_region(region, path=None):
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
    
    return gpd.read_file(DATA_PATH / f"R{region:0>2}" / "REGION_C17.shp")


def read_localidad(region, path=None, translation_path=None):
    """
    Carga la geometría de las localidades de la región indicada definidas en el censo
    2017, a partir del archivo "LOCALIDAD_C17.shp".

    Parameters
    ----------
    region: int
        Identificador numérico de la región a cargar.
    path: string, default=None
        Ubicación del archivo "LOCALIDAD_C17.shp" que contiene las geometrías.
    translation_path: string, default=None
        Ubicación del archivo "microdato_censo2017-geografia.csv" que contiene la tabla
        de traducción de comunas del censo.

    Returns
    -------
    gpd.GeoDataFrame
        Dataframe con la geometría de cada localidad de la región,
        su nombre, la localidad, comuna, provincia y región a la que pertenece,
        el área que abarca y el identificador asociado a los microdatos del censo.
    
    """
    if path is None:
        DATA_PATH = _CENSUS_MAPS
    else:
        DATA_PATH = Path(path)

    if translation_path is None:
        T_DATA_PATH = _CENSUS_PATH
    else:
        T_DATA_PATH = Path(translation_path)
    
    traduccion = pd.read_csv(T_DATA_PATH / "Tablas_parametros/microdato_censo2017-geografia.csv",  sep=";")
    shape = gpd.read_file(DATA_PATH / f"R{region:0>2}" / "LOCALIDAD_C17.shp")
    shape.COMUNA=shape.COMUNA.astype("int")
    result = shape.merge(traduccion[["ID_ZONA_LOC", 'COMUNA', 'DC', "ZC_LOC"]], left_on=['COMUNA', 'DISTRITO','LOC_ZON'], right_on=['COMUNA', 'DC', "ZC_LOC"]).drop(["DC", "ZC_LOC"],axis=1)
    return result


def read_zona(region, path=None, translation_path=None):
    """
    Carga la geometría de las zonas de la región indicada definidas en el censo
    2017, a partir del archivo "ZONA_C17.shp".

    Parameters
    ----------
    region: int
        Identificador numérico de la región a cargar.
    path: string, default=None
        Ubicación del archivo "ZONA_C17.shp" que contiene las geometrías.
    translation_path: string, default=None
        Ubicación del archivo "microdato_censo2017-geografia.csv" que contiene la tabla
        de traducción de comunas del censo.

    Returns
    -------
    gpd.GeoDataFrame
        Dataframe con la geometría de cada zona de la región,
        su nombre, la localidad, comuna, provincia y región a la que pertenece,
        el área que abarca y el identificador asociado a los microdatos del censo.
    
    """
    if path is None:
        DATA_PATH = _CENSUS_MAPS
    else:
        DATA_PATH = Path(path)
    
    if translation_path is None:
        T_DATA_PATH = _CENSUS_PATH
    else:
        T_DATA_PATH = Path(translation_path)
    
    traduccion = pd.read_csv(T_DATA_PATH/"Tablas_parametros/microdato_censo2017-geografia.csv",  sep=";")
    shape = gpd.read_file(DATA_PATH / f"R{region:0>2}" / "ZONA_C17.shp")
    shape.COMUNA=shape.COMUNA.astype("int")
    result = shape.merge(traduccion[["ID_ZONA_LOC", 'COMUNA', 'DC', "ZC_LOC"]], left_on=['COMUNA', 'DISTRITO','LOC_ZON'], right_on=['COMUNA', 'DC', "ZC_LOC"]).drop(["DC", "ZC_LOC"],axis=1)
    return result


def read_provincia(region, path=None):
    """
    Carga la geometría de las provincias de la región indicada definidas en el censo
    2017, a partir del archivo "PROVINCIA_C17.shp".

    Parameters
    ----------
    path: string, default=None
        Ubicación del archivo "PROVINCIA_C17.shp" que contiene las geometrías.

    Returns
    -------
    gpd.GeoDataFrame
        Dataframe con la geometría de cada provincia de la región,
        su nombre, la región a la que pertenece, y el área que abarca.
    
    """
    if path is None:
        DATA_PATH = _CENSUS_MAPS
    else:
        DATA_PATH = Path(path)

    return gpd.read_file(DATA_PATH / f"R{region:0>2}" / "PROVINCIA_C17.shp")


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

    src_df = df.loc[:, (col_name,)].astype(index_dtype)

    return src_df.join(values_df, on=col_name)[value_col]


def process_viviendas(
        path=None
):
    """
    Procesa el contenido del archivo "Viviendas.csv" y lo almacena en un
    archivo de formato Parquet.

    Parameters
    ----------
    path: string, default=None
        Ubicación del archivo csv con los datos de viviendas del censo 2017.
    
    """
    if path is None:
        DATA_PATH = _CENSUS_PATH
    else:
        DATA_PATH = Path(path)
    df = dd.read_csv(
        DATA_PATH / "Viviendas.csv", sep=";", decimal=",", encoding="utf-8")
    traduccion = pd.read_csv(DATA_PATH/"Tablas_parametros/microdato_censo2017-geografia.csv",  sep=";")
    cols_to_replace = ["COMUNA", "REGION", "PROVINCIA"]
    for col in cols_to_replace:
        df[col] = df['ID_ZONA_LOC'].map(traduccion.set_index('ID_ZONA_LOC')[col])
    df['NOM_COMUNA']=decode_column(df, fname=DATA_PATH/"Tablas_parametros/COMUNA.csv", index_col="COMUNA", col_name="COMUNA", sep=';')

    df.to_parquet(DATA_PATH/"Viviendas.parquet", write_index=False)


def read_viviendas(
    path=None,
    columnas=None,
    filtros=None
):
    """
    Carga el contenido del archivo "Viviendas.parquet", que contiene las respuestas sobre viviendas, a un dataframe.

    Ejemplo de uso: read_viviendas(columnas=['REGION', 'PROVINCIA', 'COMUNA', 'DC', 'AREA', 'ZC_LOC', 'ID_ZONA_LOC', 'NVIV'], filtros=[('COMUNA', "==", 13101)])

    Parameters
    ----------
    path: string, default=None
        Ubicación del archivo parquet con los datos del censo 2017.
    columnas: list, default=None
        Si no es None, solo se cargarán las columnas especificadas.
    filtros: List[Tuple] or List[List[Tuple]], default=None

    Returns
    -------
    pd.DataFrame
        Dataframe con la información sobre viviendas, con las columnas decodificadas. 
    """
    if path is None:
        DATA_PATH = _CENSUS_PATH
    else:
        DATA_PATH = Path(path)
    print(DATA_PATH)
    try:
        df = pd.read_parquet(
        DATA_PATH / "Viviendas.parquet", columns=columnas, filters=filtros)
    except FileNotFoundError:
        process_hogares(path)
        df = pd.read_parquet(
        DATA_PATH / "Viviendas.parquet", columns=columnas, filters=filtros)
    return df


def process_hogares(
        path=None
):
    """
    Procesa el contenido del archivo "Hogares.csv" y lo almacena en un
    archivo de formato Parquet.

    Parameters
    ----------
    path: string, default=None
        Ubicación del archivo csv con los datos de hogares del censo 2017.
    
    """
    if path is None:
        DATA_PATH = _CENSUS_PATH
    else:
        DATA_PATH = Path(path)
    df = dd.read_csv(
        DATA_PATH / "Hogares.csv", sep=";", decimal=",", encoding="utf-8")
    df['TIPO_OPERATIVO']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/TIPO_OPERATIVO.csv", index_col="Id", col_name="TIPO_OPERATIVO", sep=',')
    df['TIPO_HOGAR']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/TIPO_HOGAR.csv", index_col="Id", col_name="TIPO_HOGAR", sep=',')
    traduccion = pd.read_csv(DATA_PATH/"Tablas_parametros/microdato_censo2017-geografia.csv",  sep=";")
    cols_to_replace = ["COMUNA", "REGION", "PROVINCIA"]
    for col in cols_to_replace:
        df[col] = df['ID_ZONA_LOC'].map(traduccion.set_index('ID_ZONA_LOC')[col])
    df['NOM_COMUNA']=decode_column(df, fname=DATA_PATH/"Tablas_parametros/COMUNA.csv", index_col="COMUNA", col_name="COMUNA", sep=';')

    df.to_parquet(DATA_PATH/"Hogares.parquet", write_index=False)


def read_hogares(
    path=None,
    #regiones=None,
    #comunas=None,
    columnas=None,
    filtros=None
):
    """
    Carga el contenido del archivo "Hogares.parquet", que contiene las respuestas sobre hogares, a un dataframe.
    Los hogares corresponden a la manera de organización de las personas dentro de las viviendas
    particulares y constituyen por derecho propio una unidad de empadronamiento en sí mismos.

    Ejemplo de uso: read_hogares(columnas=['REGION', 'PROVINCIA', 'COMUNA', 'DC', 'AREA', 'ZC_LOC', 'ID_ZONA_LOC', 'NVIV'], filtros=[('COMUNA', "==", 13101)])

    Parameters
    ----------
    path: string, default=None
        Ubicación del archivo parquet con los datos del censo 2017.
    columnas: list, default=None
        Si no es None, solo se cargarán las columnas especificadas.
    filtros: List[Tuple] or List[List[Tuple]], default=None

    Returns
    -------
    pd.DataFrame
        Dataframe con la información sobre hogares, con las columnas decodificadas. 
    """
    if path is None:
        DATA_PATH = _CENSUS_PATH
    else:
        DATA_PATH = Path(path)
    print(DATA_PATH)
    try:
        df = pd.read_parquet(
        DATA_PATH / "Hogares.parquet", columns=columnas, filters=filtros)
    except FileNotFoundError:
        process_hogares(path)
        df = pd.read_parquet(
        DATA_PATH / "Hogares.parquet", columns=columnas, filters=filtros)
    
    return df


def process_personas(
        path=None
):
    """
    Procesa el contenido del archivo "Personas.csv" y lo almacena en un
    archivo de formato Parquet dentro del mismo directorio entregado,
    con las respuestas decodificadas. El archivo Parquet resultante
    está particionado según Región.

    Parameters
    ------------
    path: string, default=None
        Ubicación del archivo Personas.csv con los microdatos del censo 2017.
    
    """
    if path is None:
        DATA_PATH = _CENSUS_PATH
    else:
        DATA_PATH = Path(path)
    parquet_file = DATA_PATH/"Personas.parquet"
    df = dd.read_csv(DATA_PATH/"Personas.csv", sep=";")
    df['P07']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/P07.csv", index_col="id", col_name="P07", sep=',')
    df['P08']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/P08.csv", index_col="id", col_name="P08", sep=',')
    df['P10']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/P10.csv", index_col="id", col_name="P10", sep=',')
    df['P10COMUNA']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/COMUNA.csv", index_col="COMUNA", col_name="P10COMUNA", sep=';')
    df['P11COMUNA']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/COMUNA.csv", index_col="COMUNA", col_name="P11COMUNA", sep=';')
    df['P12COMUNA']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/COMUNA.csv", index_col="COMUNA", col_name="P12COMUNA", sep=';')
    df['P10PAIS']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/PAIS.csv", index_col="id", col_name="P10PAIS", sep=',')
    df['P11PAIS']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/PAIS.csv", index_col="id", col_name="P11PAIS", sep=',')
    df['P12PAIS']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/PAIS.csv", index_col="id", col_name="P12PAIS", sep=',')
    df['P11']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/P11.csv", index_col="id", col_name="P11", sep=',')
    df['P12']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/P12.csv", index_col="id", col_name="P12", sep=',')
    df['P12A_TRAMO']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/P12A_TRAMO.csv", index_col="id", col_name="P12A_TRAMO", sep=',')
    df['P13']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/P13.csv", index_col="id", col_name="P13", sep=',')
    df['P14']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/P14.csv", index_col="id", col_name="P14", sep=',')
    df['P15']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/P15.csv", index_col="id", col_name="P15", sep=',')
    df['P15A']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/P15A.csv", index_col="id", col_name="P15A", sep=',')
    df['P16']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/P16.csv", index_col="id", col_name="P16", sep=',')
    df['P16A']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/P16A.csv", index_col="id", col_name="P16A", sep=',')
    df['P16A_OTRO']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/P16A_OTRO.csv", index_col="id", col_name="P16A_OTRO", sep=',')
    df['P16A_GRUPO']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/P16A_GRUPO.csv", index_col="id", col_name="P16A_GRUPO", sep=',')
    df['P17']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/P17.csv", index_col="id", col_name="P17", sep=',')
    df['P18']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/P18.csv", index_col="id", col_name="P18", sep=',', index_dtype=str)
    df['P21M']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/P21M.csv", index_col="id", col_name="P21M", sep=',')
    df['P10PAIS_GRUPO']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/PAIS_GRUPO.csv", index_col="id", col_name="P10PAIS_GRUPO", sep=',')
    df['P11PAIS_GRUPO']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/PAIS_GRUPO.csv", index_col="id", col_name="P11PAIS_GRUPO", sep=',')
    df['P12PAIS_GRUPO']= decode_column(df, fname=DATA_PATH/"Tablas_parametros/PAIS_GRUPO.csv", index_col="id", col_name="P12PAIS_GRUPO", sep=',')
    
    traduccion = pd.read_csv(DATA_PATH/"Tablas_parametros/microdato_censo2017-geografia.csv",  sep=";")
    cols_to_replace = ["COMUNA", "REGION", "PROVINCIA"]
    for col in cols_to_replace:
        df[col] = df['ID_ZONA_LOC'].map(traduccion.set_index('ID_ZONA_LOC')[col])
    df['NOM_COMUNA']=decode_column(df, fname=DATA_PATH/"Tablas_parametros/COMUNA.csv", index_col="COMUNA", col_name="COMUNA", sep=';')

    df.to_parquet(parquet_file, write_index=False, overwrite=True, partition_on="REGION")


def read_personas(path=None, filtros=None, columnas=None):
    """
    Carga el contenido del archivo "Personas.parquet", que contiene las respuestas sobre personas, a un dataframe.
    Las personas corresponden a todos los individuos comprendidos en el censo.

    Ejemplo de uso: read_personas(columnas=['REGION', 'PROVINCIA', 'COMUNA', 'ID_ZONA_LOC', 'P08', 'P09'], filtros=[('COMUNA', "==", 13101)])

    Parameters
    ----------
    path: string, default=None
        Ubicación del archivo parquet con los datos del censo 2017.
    columnas: list, default=None
        Si no es None, solo se cargarán las columnas especificadas.
    filtros: List[Tuple] or List[List[Tuple]], default=None

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
        DATA_PATH / "Personas.parquet", columns=columnas, filters=filtros)
    except FileNotFoundError:
        process_personas(path)
        df = dd.read_parquet(
        DATA_PATH / "Personas.parquet", columns=columnas, filters=filtros)
    return df


def regions_in_geometry(geometry_df, path=None):
    """
    Retorna el listado de las regiones de Chile que intersectan la geometría entregada.

    Parameters
    ----------
    geometry_df: geopandas.dataframe
        Dataframe que contiene la geometría que se quiere analizar, en la columna "geometry".
    path: string, default=None
        Ubicación de los shapefiles del censo 2017.

    Returns
    -------
    List(int)
        Lista con los códigos numéricos de cada región presente en la geometría entregada.
    """
    #TODO: heurísitca para agilizar proceso
    intersecting_regions = []
    for i in range(1, 17):
        df_region = read_region(i, path=path)
        overlap = geometry_df.to_crs(df_region.crs).unary_union.intersects(df_region.geometry)
        #spatial_index2 = df_region.sindex
        #overlap = zoning.geometry.apply(lambda x: any(spatial_index2.intersection(x.bounds)))
        if overlap.any():
            intersecting_regions.append(i)
    return intersecting_regions


def overlay_zoning(zoning):
    """
    Segmenta la zonificación del censo 2017 a partir de una zonificación entregada, generando una nueva geometría
    a partir de la superposición de ambas.

    En el notebook `notebooks/census-wrapper/demography_by_zoning.ipynb` se encuentra un ejemplo de uso de esta función.

    Parameters
    ----------
    zoning: geopandas.dataframe
        Dataframe que contiene la zonificación que se intersectará con la del censo, en la columna `geometry`.
    
    Returns
    -------
    geopandas.dataframe
        Dataframe que contiene la geometría resultante de la intersección. Este dataframe incluye
        la columna `percentage_overlap` que indica la proporción de área censal cubierta por cada geometría
        resultante.
    """
    # determine regions to load
    regiones = regions_in_geometry(zoning)
    # load and concat census geometry from regiones
    censo_zoning_by_region = []
    for r in regiones:
        censo_zoning_by_region.append(read_zona(r))
        censo_zoning_by_region.append(read_localidad(r))
    censo_zoning = pd.concat(censo_zoning_by_region)
    censo_zoning.COMUNA=censo_zoning.COMUNA.astype("int")
    censo_zoning['area_censo'] = censo_zoning.to_crs('EPSG:3857').area
    # overlay two zonings, convert crs
    merged_zoning =  gpd.overlay(zoning.to_crs('EPSG:3857'),censo_zoning.to_crs('EPSG:3857'), how='intersection',keep_geom_type=False)
    merged_zoning['area_m2'] = merged_zoning.to_crs('EPSG:3857').area
    # Add column indicating how much of the censal area is covered
    merged_zoning['percentage_overlap'] = (merged_zoning.area_m2 / merged_zoning.area_censo).fillna(0)
    # Delete insignificant areas
    merged_zoning = merged_zoning[merged_zoning['area_m2'] > 100000]
    return merged_zoning


def aggregate_by_zoning(zoning, df, df_id, columns, zoning_unique_id):
    census_data = zoning.merge(df, on=df_id)
    for label in columns:
        census_data[label] = census_data[label] * census_data['percentage_overlap']
    grouped_by_zone = census_data.groupby(zoning_unique_id)[columns].sum()
    grouped_by_zone.reset_index(inplace=True)
    return grouped_by_zone


def population_by_zoning(intersected_zoning, zoning_unique_id):
    """
    Obtiene un estimado de la cantidad de personas en una zonificación geográfica.

    En el notebook demography_by_zoning.ipynb se encuentra un ejemplo de uso de esta función.

    Parameters
    ----------
    intersected_zoning: geopandas.dataframe
        Dataframe que contiene la zonificación objetivo intersectada con la del censo,
        resultante de llamar la función :func:`~aves.data.census.overlay_zoning`.
    zoning_unique_id: int or String
        Nombre de la columna de `intersected_zoning` que contiene el identificador único
        de cada zona.
    
    Returns
    -------
    pandas.dataframe
        Dataframe que contiene el estimado de población por zona.
    """
    region_list = intersected_zoning.REGION.unique()
    # get personal_data
    df = read_personas(columnas=['ID_ZONA_LOC', 'PERSONAN'], filtros=[("REGION", "in", region_list)])
    agg_count = pd.NamedAgg(column='PERSONAN', aggfunc="count")
    reduced_df = df.groupby("ID_ZONA_LOC").agg(poblacion=agg_count)
    grouped_by_zone = aggregate_by_zoning(intersected_zoning, reduced_df, "ID_ZONA_LOC", ['poblacion'], zoning_unique_id)
    
    return grouped_by_zone


def sex_by_zoning(intersected_zoning, zoning_unique_id):
    """
    Obtiene la distribución por sexo de la población segmentada en una zonificación geográfica.

    En el notebook demography_by_zoning.ipynb se encuentra un ejemplo de uso de esta función.

    Parameters
    ----------
    intersected_zoning: geopandas.dataframe
        Dataframe que contiene la zonificación objetivo intersectada con la del censo,
        resultante de llamar la función :func:`~aves.data.census.overlay_zoning`.
    zoning_unique_id: int or String
        Nombre de la columna de `intersected_zoning` que contiene el identificador único
        de cada zona.
    
    Returns
    -------
    pandas.dataframe
        Dataframe que contiene el estimado de población masculina y femenina por cada zona.
    """
    region_list = intersected_zoning.REGION.unique()
    sex_by_region = read_personas(columnas=['ID_ZONA_LOC', 'P08'], filtros=[("REGION", "in", region_list)])
    reduced_df = sex_by_region.groupby(["ID_ZONA_LOC"])['P08'].value_counts().unstack(fill_value=0)
    reduced_df.reset_index(inplace=True)
    grouped_by_zone = aggregate_by_zoning(intersected_zoning, reduced_df, "ID_ZONA_LOC", ["Mujer", "Hombre"], zoning_unique_id)
    return grouped_by_zone


def age_by_zoning(intersected_zoning, zoning_unique_id):
    """
    Obtiene la distribución por edad de la población segmentada en una zonificación geográfica.

    En el notebook demography_by_zoning.ipynb se encuentra un ejemplo de uso de esta función.

    Parameters
    ----------
    intersected_zoning: geopandas.dataframe
        Dataframe que contiene la zonificación objetivo intersectada con la del censo,
        resultante de llamar la función :func:`~aves.data.census.overlay_zoning`.
    zoning_unique_id: int or String
        Nombre de la columna de `intersected_zoning` que contiene el identificador único
        de cada zona.
    
    Returns
    -------
    pandas.dataframe
        Dataframe que contiene el estimado de población correspondiente a cada grupo etario por zona.
    """
    # TODO: incluir parámetro de binning
    region_list = intersected_zoning.REGION.unique()
    age_by_region = read_personas(columnas=['ID_ZONA_LOC', 'P09'], filtros=[("REGION", "in", region_list)])
    bins = [0, 18, 65, 100, 101]
    labels = ["Menor de edad","Mayor de edad","Adulto mayor", "100 años y más"]
    age_by_region['age_group'] = pd.cut(age_by_region['P09'], bins=bins, labels=labels, include_lowest=True, right=False)
    reduced_df = age_by_region.groupby(["ID_ZONA_LOC"])['age_group'].value_counts().unstack(fill_value=0)
    reduced_df.reset_index(inplace=True)
    grouped_by_zone = aggregate_by_zoning(intersected_zoning, reduced_df, "ID_ZONA_LOC", labels, zoning_unique_id)

    return grouped_by_zone


def inmigrants_by_zoning(intersected_zoning, zoning_unique_id):
    """
    Obtiene un estimado de la cantidad de migrantes en una zonificación geográfica.

    En el notebook demography_by_zoning.ipynb se encuentra un ejemplo de uso de esta función.

    Parameters
    ----------
    intersected_zoning: geopandas.dataframe
        Dataframe que contiene la zonificación objetivo intersectada con la del censo,
        resultante de llamar la función :func:`~aves.data.census.overlay_zoning`.
    zoning_unique_id: int or String
        Nombre de la columna de `intersected_zoning` que contiene el identificador único
        de cada zona.
    
    Returns
    -------
    pandas.dataframe
        Dataframe que contiene el estimado de población migrante por zona.
    """
    # TODO: incluir zonas con 0
    region_list = intersected_zoning.REGION.unique()
    # get personal_data
    inmigrants_by_region = read_personas(columnas=['ID_ZONA_LOC', 'P10','P12'], filtros=[("REGION", "in", region_list)])
    condition_birth = ~inmigrants_by_region['P12'].isin(['Missing', "En esta comuna", "En otra comuna"])
    condition_residence = inmigrants_by_region['P10'] != 'En otro país'
    inmigrants_by_region = inmigrants_by_region[condition_birth & condition_residence]
    agg_count = pd.NamedAgg(column='P12', aggfunc="count")
    reduced_df = inmigrants_by_region.groupby("ID_ZONA_LOC").agg(inmigrantes=agg_count)
    grouped_by_zone = aggregate_by_zoning(intersected_zoning, reduced_df, "ID_ZONA_LOC", ['inmigrantes'], zoning_unique_id)
    
    return grouped_by_zone


def schooling_by_zoning(intersected_zoning, zoning_unique_id):
    """
    Obtiene la distribución por años de escolaridad de la población segmentada en una zonificación geográfica.

    En el notebook demography_by_zoning.ipynb se encuentra un ejemplo de uso de esta función.

    Parameters
    ----------
    intersected_zoning: geopandas.dataframe
        Dataframe que contiene la zonificación objetivo intersectada con la del censo,
        resultante de llamar la función :func:`~aves.data.census.overlay_zoning`.
    zoning_unique_id: int or String
        Nombre de la columna de `intersected_zoning` que contiene el identificador único
        de cada zona.
    
    Returns
    -------
    pandas.dataframe
        Dataframe que contiene el estimado de población y años de escolaridad por cada zona.
    """
    region_list = intersected_zoning.REGION.unique()
    schooling_by_region = read_personas(columnas=['ID_ZONA_LOC', 'ESCOLARIDAD'], filtros=[("REGION", "in", region_list)])
    columns = list(schooling_by_region.ESCOLARIDAD.unique())
    reduced_df = schooling_by_region.groupby(["ID_ZONA_LOC"])['ESCOLARIDAD'].value_counts().unstack(fill_value=0)
    reduced_df.reset_index(inplace=True)
    grouped_by_zone = aggregate_by_zoning(intersected_zoning, reduced_df, "ID_ZONA_LOC", columns, zoning_unique_id)
    return grouped_by_zone


def indigenous_by_zoning(intersected_zoning, zoning_unique_id):
    """
    Obtiene un estimado de la cantidad de personas que se identifican como pertenecientes a un
    pueblo originario en una zonificación geográfica.

    En el notebook demography_by_zoning.ipynb se encuentra un ejemplo de uso de esta función.

    Parameters
    ----------
    intersected_zoning: geopandas.dataframe
        Dataframe que contiene la zonificación objetivo intersectada con la del censo,
        resultante de llamar la función :func:`~aves.data.census.overlay_zoning`.
    zoning_unique_id: int or String
        Nombre de la columna de `intersected_zoning` que contiene el identificador único
        de cada zona.
    
    Returns
    -------
    pandas.dataframe
        Dataframe que contiene el estimado de población por zona.
    """
    region_list = intersected_zoning.REGION.unique()
    # get personal_data
    df = read_personas(columnas=['ID_ZONA_LOC', 'P16'], filtros=[("REGION", "in", region_list)])
    df = df[df['P16']=='Sí']
    agg_count = pd.NamedAgg(column='P16', aggfunc="count")
    reduced_df = df.groupby("ID_ZONA_LOC").agg(pueblo_originario=agg_count)
    reduced_df.reset_index(inplace=True)
    grouped_by_zone = aggregate_by_zoning(intersected_zoning, reduced_df, "ID_ZONA_LOC", ['pueblo_originario'], zoning_unique_id)
    
    return grouped_by_zone

"""
def aggregate_by_zoning(zoning, criteria):
    
    #TODO: load person data, define criteria
    columns = ["NOM_COMUNA", 'REGION', 'PROVINCIA', 'COMUNA', 'DC', 'AREA', 'ZC_LOC', 'ID_ZONA_LOC']
    columns.append(criteria)
    census_data = read_personas(filtros=[("REGION", "in", regiones)], columnas=columns)
    #TODO: merge census data with census geometry
    census_data_geo = censo_zoning.merge(census_data, left_on=['COMUNA', 'DISTRITO','LOC_ZON'], 
                                         right_on=['COMUNA', 'DC', "ZC_LOC"], suffixes=["map", None]
                                         )

    #TODO: group by zoning and aggregate criteria
    res = merged_zoning.groupby('right_index').agg({criteria[0]: 'count'})

    grouped_data_geo = zoning.merge(res, left_on='right_index', right_index=True)

    df = None
    return df"""



