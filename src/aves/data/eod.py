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
    ----------
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
        Identificación del _character set_ que utiliza el archivo. Usualmente es utf-8, si no funciona,
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


def read_trips(
    path=None, decode_columns=True, remove_invalid=True, fix_clock_times=True
):
    """ 
    Lee los archivos que contienen los resultados de la encuesta origen destino y crea una tabla con
    la información de los viajes.

    Parameters
    ----------
    path : string, default=None
        Ruta del archivo que contiene los viajes.
    decode_column: bool, default=True
        Indica si se quiere decodificar el contenido de las columnas, reemplazando IDs por su significado
    remove_invalid: bool, default=True
        Indica si se quiere eliminar filas que no tienen hora o que han sido inputadas.
    fix_clock_times: bool, default=True
        Indica si se desea estandarizar la hora de inicio al formato timedelta.

    Returns
    -------
    pd.DataFrame
        Dataframe con la información de viajes de la encuesta origen-destino. Las columnas son las siguientes:

        ==========  ==============================================================
        Hogar
		Persona
		Viaje
		Etapas
		ComunaOrigen
		ComunaDestino
        SectorOrigen
		SectorDestino
		ZonaOrigen
		ZonaDestino
        OrigenCoordX
		OrigenCoordY
		DestinoCoordX
		DestinoCoordY
        Proposito
		PropositoAgregado
		ActividadDestino
		MediosUsados
        ModoAgregado
		ModoPriPub
		ModoMotor
		HoraIni
		HoraFin
        HoraMedia
		TiempoViaje
		TiempoMedio
		Periodo
		MinutosDespues
        CuadrasDespues
		FactorLaboralNormal
		FactorSabadoNormal
        FactorDomingoNormal
		FactorLaboralEstival
        FactorFindesemanaEstival
		CodigoTiempo
		ModoDifusion
        DistEuclidiana
		DistManhattan
		Imputada
        ==========  ==============================================================
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

    if fix_clock_times:
        df["HoraIni"] = pd.to_timedelta(df["HoraIni"] + ":00")
    return df


def read_homes(path=None):
    """

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
    if path is None:
        DATA_PATH = _EOD_MAPS
    else:
        DATA_PATH = Path(path)

    return gpd.read_file(DATA_PATH)


def read_vehicles(path=None, decode_columns=True):
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
            index_dtype=str,
            encoding="iso-8859-1",
        )

    return df
