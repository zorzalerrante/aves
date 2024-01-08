def read_concat_date(start_date, end_date, data_path):
    """
    Recibe la fecha desde la cual se quiere leer los viajes (start_date) hasta (end_date),
    los concatena y devuelve el DataFrame resultante.

    Parameters
    ------------
    start_date: string
        Fecha de inicio en formato 'YYYY-MM-DD'.
    end_date: string
        Fecha de fin en formato 'YYYY-MM-DD'.
    data_path: Path
        Ruta al directorio que contiene los archivos a leer.

    Returns
    -------
    dask.dataframe.DataFrame
        DataFrame concatenado resultante.
    """
    # Genera las fechas entre start_date y end_date
    date_range = [datetime.strftime(d, "%Y-%m-%d") for d in pd.date_range(start=start_date, end=end_date)]

    # Lee los dataframes y agrega una columna de fecha
    dataframes = []
    for fecha in date_range:
        parquet_path = data_path / f"{fecha}.parquet"
        csv_path = data_path / f"{fecha}.csv"
        gz_path = data_path / f"{fecha}.gz"

        if parquet_path.exists():
            df = dd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = dd.read_csv(csv_path)
        elif gz_path.exists():
            df = dd.read_csv(gz_path, compression='gzip')
        else:
            raise FileNotFoundError(f"No se encontraron archivos para la fecha {fecha}")

        df['fecha'] = datetime.strptime(fecha, "%Y-%m-%d")
        dataframes.append(df)

    # Concatena los dataframes a lo largo del eje de las filas
    df_concatenado = dd.concat(dataframes, axis=0)

    df_concatenado = df_concatenado.assign(
        hour=lambda x: x["fecha"].dt.hour,
        dayofweek=lambda x: x["fecha"].dt.dayofweek,
        tiempo=1
    )

    return df_concatenado
