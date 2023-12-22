
from shapely.geometry import LineString, Point

# Función para obtener el punto de una LineString
def get_point(line, index=0):
    return Point(line.xy[0][index], line.xy[1][index])

def calcular_treatment_zones(scooters_path, zones):
    """
    Procesa los datos de viajes de scooters y devuelve un DataFrame con el conteo de viajes por zona.

    Parameters:
    - scooters_path (str): Ruta al archivo geojson que contiene los datos de viajes de scooters.
    - zones (geopandas.GeoDataFrame): GeoDataFrame que representa las zonas de tratamiento.

    Returns:
    - pandas.DataFrame: DataFrame resultante con el conteo de viajes de scooters por zona.
    """
    # Lee datos de viajes de scooters
    scooter_trips = gpd.read_file(scooters_path)
    
    # Convierte el atributo departure_time a formato datetime
    scooter_trips["departure_time"] = pd.to_datetime(
        scooter_trips["departure_time"], format="mixed", dayfirst=False
    )

    # Asigna nuevas columnas al DataFrame de viajes de scooters
    scooter_trip_origins = scooter_trips.assign(
        hour=lambda x: x["departure_time"].dt.hour,
        dayofweek=lambda x: x["departure_time"].dt.dayofweek,
        year=lambda x: x["departure_time"].dt.year,
        tiempo=1,
        geometry=scooter_trips.geometry.map(lambda x: get_point(x, 0))
    )

    # Realiza un join y calcula el conteo de viajes por zona
    treatment_by_zones = (
        gpd.sjoin(scooter_trip_origins, zones, predicate="within")
        .groupby(['year', 'dayofweek', "hour", "ZONA777"])
        .size()
        .rename("scooter_trips")
        .reset_index()
        .assign(tiempo=1)
        .rename(columns={"ZONA777": "diseno777subida"})
    )

    return treatment_by_zones

"""
En el caso de estudio se usa:
treatment_zones = calcular_treatment_zones(scooters_path, zones)
"""