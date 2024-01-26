from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

_DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data"
_EOD_PATH = _DATA_PATH / "external" / "EOD_STGO"
_EOD_MAPS = _DATA_PATH / "external" / "Zonificacion_EOD2012"


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

#Funcion para asignar tipo de día
def etiquetar_tipo_dia(row):
    if pd.notna(row['FactorLaboralNormal']):
        return 'Laboral'
    if pd.notna(row['FactorDomingoNormal']):
        return 'Domingo'
    if pd.notna(row['FactorSabadoNormal']):
        return 'Sábado'
    if pd.notna(row['FactorLaboralEstival']):
        return 'LaboralEstival'
    if pd.notna(row['FactorFindesemanaEstival']):
        return 'FindesemanaEstival'
    else:
        return 'No Definido'

#Funcion para asignar Factor Externo
def etiquetar_FactorExp(row):
    if pd.notna(row['FactorLaboralNormal']):
        return row['FactorLaboralNormal']
    if pd.notna(row['FactorDomingoNormal']):
        return row['FactorDomingoNormal']
    if pd.notna(row['FactorSabadoNormal']):
        return row['FactorSabadoNormal']
    if pd.notna(row['FactorLaboralEstival']):
        return row['FactorLaboralEstival']
    if pd.notna(row['FactorFindesemanaEstival']):
        return row['FactorFindesemanaEstival']
    else:
        return None
    
def read_trips(
    path=None, decode_columns=True, remove_invalid=True, fix_clock_times=True
):
    """ 
    Busca los archivos "viajes.csv", "ViajesDifusion.csv" y "DistanciaViaje.csv" dentro
    del directorio especificado o en su defecto en el definido en la variable global "_EOD_PATH".
    Unifica la información de estos archivos en un dataframe de pandas.
    En el notebook ubicado en notebooks/gds-course/01-scl-travel-survey-maps.ipynb se pueden encontrar ejemplos de uso
    de esta función.

    Parameters
    ----------
    path : string, default=None
        Ubicación de los archivos csv con la data de la encuesta origen destino.
    decode_column: bool, default=True
        Indica si se quiere decodificar el contenido de las columnas, reemplazando IDs por su significado
        según las tablas de decodificación ubicadas en el directorio "Tablas_parametros".
    remove_invalid: bool, default=True
        Indica si se quiere eliminar filas que no tienen hora o que han sido inputadas.
    fix_clock_times: bool, default=True
        Indica si se desea estandarizar la hora de inicio al formato timedelta.

    Returns
    -------
    pd.DataFrame
        Dataframe con la información de viajes de la encuesta origen-destino.
    """
    if path is None:
        DATA_PATH = _EOD_PATH
    else:
        DATA_PATH = Path(path)

    df = (
        pd.read_csv(DATA_PATH / "viajes.csv", sep=";", decimal=",")
        .join(
            pd.read_csv(DATA_PATH / "ViajesDifusion.csv", sep=";", index_col="Viaje"),
            on="Viaje",
        )
        .join(
            pd.read_csv(DATA_PATH / "DistanciaViaje.csv", sep=";", index_col="Viaje"),
            on="Viaje",
        )
    )

    if decode_columns:
        df["ModoAgregado"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "ModoAgregado.csv",
            "ModoAgregado",
            index_col="ID",
            value_col="Modo",
        )
        df["ModoDifusion"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "ModoDifusion.csv",
            "ModoDifusion",
            encoding="latin-1",
            index_col="ID",
        )
        df["SectorOrigen"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "Sector.csv",
            col_name="SectorOrigen",
            index_col="Sector",
            value_col="Nombre",
            sep=";",
        )
        df["SectorDestino"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "Sector.csv",
            col_name="SectorDestino",
            index_col="Sector",
            value_col="Nombre",
            sep=";",
        )
        df["Proposito"] = decode_column(
            df, DATA_PATH / "Tablas_parametros" / "Proposito.csv", col_name="Proposito"
        )
        df["ComunaOrigen"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "Comunas.csv",
            "ComunaOrigen",
            value_col="Comuna",
            sep=",",
        )
        df["ComunaDestino"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "Comunas.csv",
            "ComunaDestino",
            value_col="Comuna",
            sep=",",
        )
        df["ActividadDestino"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "ActividadDestino.csv",
            "ActividadDestino",
        )
        df["Periodo"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "Periodo.csv",
            "Periodo",
            sep=";",
            value_col="Periodos",
        )

    if remove_invalid:
        df = df[pd.notnull(df["HoraIni"])]
        df = df[df["Imputada"] == 0].copy()
        df = df[df["DistManhattan"] != -1].copy()

    if fix_clock_times:
        df["HoraIni"] = pd.to_timedelta(df["HoraIni"] + ":00")

    # Aplicamos la función a cada fila y creamos una nueva columna llamada 'TipoDia'
    df['TipoDia'] = df.apply(etiquetar_tipo_dia, axis=1)

    df['FactorExpansion'] = df.apply(etiquetar_FactorExp, axis=1)

    return df


def read_homes(path=None):
    """
    Carga el contenido del archivo "Hogares.csv", que contiene las respuestas sobre hogares participantes de la
    encuesta origen destino, a un dataframe.
    En el notebook ubicado en notebooks/gds-course/01-scl-travel-survey-maps.ipynb se pueden encontrar ejemplos de uso
    de esta función.

    Parameters
    ----------
    path: string, default=None
        Ubicación de los archivos csv con la data de la encuesta origen destino.

    Returns
    -------
    pd.DataFrame
        Dataframe con la información sobre hogares, con las columnas decodificadas.
    
    """
    if path is None:
        DATA_PATH = _EOD_PATH
    else:
        DATA_PATH = Path(path)

    df = pd.read_csv(
        DATA_PATH / "Hogares.csv", sep=";", decimal=",", encoding="utf-8"
    ).rename(columns={"Factor": "FactorHogar"})

    df["Sector"] = decode_column(
        df, DATA_PATH / "Tablas_parametros" / "Sector.csv", "Sector"
    )

    return df


def read_people(path=None, decode_columns=True):
    """
    Carga el contenido del archivo "personas.csv", que contiene información sobre las personas encuestadas, a un dataframe.
    En el notebook ubicado en notebooks/gds-course/01-scl-travel-survey-maps.ipynb se pueden encontrar ejemplos de uso
    de esta función.

    Parameters
    ----------
    path: string, default=None
        Ubicación de los archivos csv con la data de la encuesta origen destino.
    decode_columns: bool, default=True
        Indica si se quiere decodificar el contenido de las columnas, reemplazando IDs por su significado

    Returns
    -------
    pd.DataFrame
        Dataframe con la información sobre personas.
    
    """
    if path is None:
        DATA_PATH = _EOD_PATH
    else:
        DATA_PATH = Path(path)

    df = pd.read_csv(
        DATA_PATH / "personas.csv", sep=";", decimal=",", encoding="utf-8"
    ).rename(columns={"Factor": "FactorPersona"})

    if decode_columns:
        df["Sexo"] = decode_column(
            df, DATA_PATH / "Tablas_parametros" / "Sexo.csv", "Sexo"
        )
        df["TramoIngreso"] = decode_column(
            df, DATA_PATH / "Tablas_parametros" / "TramoIngreso.csv", "TramoIngreso"
        )
        df["Relacion"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "Relacion.csv",
            "Relacion",
            value_col="relacion",
        )
        df["Ocupacion"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "Ocupacion.csv",
            "Ocupacion",
            value_col="ocupacion",
        )
    return df


def read_transantiago_usage(path=None, decode_columns=True):
    """
    Crea un dataframe que contiene a las personas que no usaron el sistema Transantiago en su viaje y la razón.

    Parameters
    ----------
    path: string, default=None
        Ubicación de los archivos csv con la data de la encuesta origen destino.
    decode_columns: bool, default=True
        Indica si se quiere decodificar el contenido de las columnas, reemplazando IDs por su significado

    Returns
    -------
    pd.DataFrame
        Dataframe con una fila por persona que no usó el Transantiago y su razón para no hacerlo.
    
    """
    if path is None:
        DATA_PATH = _EOD_PATH
    else:
        DATA_PATH = Path(path)

    df = (
        pd.read_csv(DATA_PATH / "personas.csv", sep=";", decimal=",", encoding="utf-8")
        .pipe(lambda x: x[pd.notnull(x.NoUsaTransantiago)])
        .set_index("Persona")["NoUsaTransantiago"]
        .str.split(";")
        .explode()
        .reset_index()
    )

    if decode_columns:
        df["NoUsaTransantiago"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "NoUsaTransantiago.csv",
            "NoUsaTransantiago",
            index_dtype=str,
        )

    return df


def read_zone_design(path=None):
    """
    Carga la geometría de la zonificación de las comunas participantes en la encuesta.
    Podemos encontrar un tutorial de uso de esta función en el notebook ` notebooks/vis-course/03-python-mapas-preliminario.ipynb` 

    Parameters
    ----------
    path: string, default=None
        Ubicación del archivo shapefile que contiene la geometría de las comunas. Si no se especifica, se usará el valor
        almacenado en la variable global _EOD_MAPS.
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataframe con la geometría de la zonificación de las comunas, el sistema de coordenadas usado es [`EPSG:32719`](https://epsg.io/32719)
    
    """
    
    if path is None:
        DATA_PATH = _EOD_MAPS
    else:
        DATA_PATH = Path(path)

    return gpd.read_file(DATA_PATH)


def read_vehicles(path=None, decode_columns=True):
    """
    Carga el contenido del archivo "Vehiculo.csv", que contiene información sobre los vehículos de los
    hogares encuestados.

    Parameters
    ----------
    path: string, default=None
        Ubicación de los archivos csv con la data de la encuesta origen destino.
    decode_columns: bool, default=True
        Indica si se quiere decodificar el contenido de las columnas, reemplazando IDs por su significado

    Returns
    -------
    pd.DataFrame
        Dataframe con la información sobre personas.
    
    """
    if path is None:
        DATA_PATH = _EOD_PATH
    else:
        DATA_PATH = Path(path)

    df = pd.read_csv(
        DATA_PATH / "Vehiculo.csv", sep=";", decimal=",", encoding="iso-8859-1"
    )

    if decode_columns:
        df["TipoVeh"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "TipoVeh.csv",
            value_col="vehiculo",
            col_name="TipoVeh",
            index_dtype=int,
            encoding="iso-8859-1",
        )

    return df
