
from aves.features.osm.pois import explode_tags

def calcular_zone_metro_nearby(osm, zones, SCOOTERS_PATH):
    """
    Procesa los datos de estaciones de metro, red ferroviaria y zonas cercanas al metro.

    Parameters:
    - osm: Objeto que proporciona acceso a datos de OpenStreetMap.
    - zones (geopandas.GeoDataFrame): GeoDataFrame que representa las zonas de tratamiento.
    - SCOOTERS_PATH (str): Ruta al archivo geojson que contiene los datos de viajes de scooters.

    Returns:
    - pandas.DataFrame: DataFrame resultante con los datos procesados para el análisis de regresión.
    """
    # Obtiene estaciones de metro
    metro_stations = osm.get_data_by_custom_criteria(
        custom_filter={"public_transport": ["station"]},
        keep_ways=False,
        keep_relations=False,
        keep_nodes=True,
    ).pipe(lambda x: x[x["operator"].isin(["Metro S.A.", "Tren Central"])])

    # Obtiene red ferroviaria
    rail_network = (
        osm.get_data_by_custom_criteria(
            custom_filter=dict(railway=["subway", "usage", "rail"]),
            osm_keys_to_keep=["railway"],
            filter_type="keep",
        )
        .pipe(explode_tags)
        .pipe(lambda x: x[x["name"].fillna("").str.contains("Línea|Metro")])
    )

    # Crea buffer de área de la red ferroviaria
    network_buffer_area = rail_network.assign(
        geometry=lambda x: x.to_crs("epsg:5361").buffer(750).to_crs("epsg:4326")
    )

    # Identifica zonas cercanas al metro
    zone_metro_nearby = (
        gpd.sjoin(zones, network_buffer_area[["name", "geometry"]], predicate="intersects")
        .set_index("ZONA777")
        .pipe(lambda x: pd.get_dummies(x["name"]).astype(int))
        .reset_index()
        .groupby("ZONA777")
        .sum()
        .astype(bool)
        .astype(int)
    )
    
    zone_metro_nearby.columns = list(
        map(lambda x: x.replace("Línea ", "L").replace(" ", ""), zone_metro_nearby.columns)
    )

    return zone_metro_nearby

"""
En el caso de estudio se usa:

#creación del objetos osm
OSM_PATH = Path('/home/nat/Escritorio/viejoAves/aves/data/external/OSM')
osm_clipped_file = OSM_PATH / 'clipped-scl-osm.pbf'
import pyrosm
osm = pyrosm.OSM(str(osm_clipped_file))
osm_clipped_file = OSM_PATH / 'clipped-scl-osm.pbf'
if not osm_clipped_file.exists():
    import os
    bounds = zones.total_bounds
    print(bounds)
    print(f"osmconvert {OSM_PATH / 'chile-latest.osm.pbf'} -b={bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]} -o={osm_clipped_file}")
    os.system(f"osmconvert {OSM_PATH / 'chile-latest.osm.pbf'} -b={bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]} -o={osm_clipped_file}")
else:
    print('data already available :D')

#Uso de la función:
zone_metro_nearby = calcular_zone_metro_nearby(osm, zones, SCOOTERS_PATH)

"""