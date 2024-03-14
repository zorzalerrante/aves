from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import dask.dataframe as dd

import aves.data.census.loading as loading

_DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data"
_CENSUS_MAPS = _DATA_PATH / "external" / "censo_2017/geometria"
_CENSUS_PATH = _DATA_PATH / "external" / "censo_2017"


def get_urban_zones(zones, path=None, remove_artifacts=True, area_threshold=100000):
    """
    Filtra una zona geográfica, retornando aquellas zonas que están dentro del límite urbano censal según
    el Censo 2017.

    Parameters
    ----------
    zones: geopandas.dataframe
        Zona geográfica a filtrar.
    path: string, default=None
        Ubicación de los archivos de contienen los shapefiles del censo.
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
    regions = regions_in_geometry(zones, path=path)
    zoning_by_region = []
    for r in regions:
        zoning_by_region.append(loading.read_limite_urbano_censal(r))
    urban_zones = pd.concat(zoning_by_region)
    clipped_zones = gpd.overlay(zones, urban_zones.to_crs(zones.crs), how='intersection')
    if remove_artifacts:
        clipped_zones['area_m2'] = clipped_zones.area
        clipped_zones = clipped_zones[clipped_zones['area_m2'] > area_threshold]
    return clipped_zones


def get_urban_municipalities(region, path=None):
    """
    Filtra una zona geográfica, retornando aquellas zonas que están dentro del límite urbano censal según
    el Censo 2017.

    Parameters
    ----------
    region: geopandas.dataframe
        Región de Chile que se desea analizar
    path: string, default=None
        Ubicación de los archivos de contienen los shapefiles del censo.

    Returns
    -------
    List
        Lista 
    """
    comunas = loading.read_comuna(region, path=path)
    zones = loading.read_limite_urbano_censal(region, path=path).to_crs(comunas.crs)
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
        df_region = loading.read_region(i, path=path)
        overlap = geometry_df.to_crs(df_region.crs).unary_union.intersects(df_region.geometry)
        #spatial_index2 = df_region.sindex
        #overlap = zoning.geometry.apply(lambda x: any(spatial_index2.intersection(x.bounds)))
        if overlap.any():
            intersecting_regions.append(i)
    return intersecting_regions


def overlay_zoning(zoning, crs='EPSG:3857', area_threshold=100000, path=None):
    """
    Segmenta la zonificación del censo 2017 a partir de una zonificación entregada, generando una nueva geometría
    a partir de la superposición de ambas.

    En el notebook `notebooks/census-wrapper/demography_by_zoning.ipynb` se encuentra un ejemplo de uso de esta función.

    Parameters
    ----------
    zoning: geopandas.dataframe
        Dataframe que contiene la zonificación que se intersectará con la del censo, en la columna `geometry`.
    crs: string, default="ESPG:3857
        Sistema de coordenadas que se usará al calcular el área cubierta por las geometrías.
    area_threshold: int, default=100000
        Umbral para filtrar artefactos resultantes de la inversección de la zonificación. Cualquier
        geometría cuya área sea menor a este valor será eliminada del resultado. Se expresa en metros cuadrados.
    path: string, default=None
        Ubicación de los shapefiles que contienen las geometrías del censo.
        
    Returns
    -------
    geopandas.dataframe
        Dataframe que contiene la geometría resultante de la intersección. Este dataframe incluye
        la columna `percentage_overlap` que indica la proporción de área censal cubierta por cada geometría
        resultante.
    """
    # determine regions to load
    regiones = regions_in_geometry(zoning, path=path)
    # load and concat census geometry from regiones
    censo_zoning_by_region = []
    for r in regiones:
        censo_zoning_by_region.append(loading.read_zona(r, path=path))
        censo_zoning_by_region.append(loading.read_localidad(r, path=path))
    censo_zoning = pd.concat(censo_zoning_by_region)
    censo_zoning.COMUNA=censo_zoning.COMUNA.astype("int")
    censo_zoning['area_censo'] = censo_zoning.to_crs(crs).area
    # overlay two zonings, convert crs
    merged_zoning =  gpd.overlay(zoning.to_crs(crs),censo_zoning.to_crs(crs), how='intersection',keep_geom_type=False)
    merged_zoning['area_m2'] = merged_zoning.to_crs(crs).area
    # Add column indicating how much of the censal area is covered
    merged_zoning['percentage_overlap'] = (merged_zoning.area_m2 / merged_zoning.area_censo).fillna(0)
    # Delete insignificant areas
    merged_zoning = merged_zoning[merged_zoning['area_m2'] > area_threshold]
    return merged_zoning


def aggregate_by_zoning(zoning, df, df_id, columns, zoning_unique_id, agg_function=sum):
    census_data = zoning.merge(df, on=df_id)
    #TODO cambiar para que reciba funciones de agregación 
    for label in columns:
        census_data[label] = census_data[label] * census_data['percentage_overlap']
    grouped_by_zone = census_data.groupby(zoning_unique_id)[columns].sum()
    grouped_by_zone.reset_index(inplace=True)
    return grouped_by_zone


def population_by_zoning(intersected_zoning, zoning_unique_id, path=None):
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
    path: string, default=None
        Ubicación del archivo "Personas.parque" o "Personas.csv" que contienen
        las respuestas del censo.
    
    Returns
    -------
    pandas.dataframe
        Dataframe que contiene el estimado de población por zona.
    """
    region_list = intersected_zoning.REGION.unique()
    # get personal_data
    df = loading.read_personas(columnas=['ID_ZONA_LOC', 'PERSONAN'], filtros=[("REGION", "in", region_list)], path=path)
    agg_count = pd.NamedAgg(column='PERSONAN', aggfunc="count")
    reduced_df = df.groupby("ID_ZONA_LOC").agg(poblacion=agg_count)
    grouped_by_zone = aggregate_by_zoning(intersected_zoning, reduced_df, "ID_ZONA_LOC", ['poblacion'], zoning_unique_id)
    
    return grouped_by_zone


def sex_by_zoning(intersected_zoning, zoning_unique_id, path=None):
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
    path: string, default=None
        Ubicación del archivo "Personas.parque" o "Personas.csv" que contienen
        las respuestas del censo.
    
    Returns
    -------
    pandas.dataframe
        Dataframe que contiene el estimado de población masculina y femenina por cada zona.
    """
    region_list = intersected_zoning.REGION.unique()
    sex_by_region = loading.read_personas(columnas=['ID_ZONA_LOC', 'P08'], filtros=[("REGION", "in", region_list)], path=path)
    reduced_df = sex_by_region.groupby(["ID_ZONA_LOC"])['P08'].value_counts().unstack(fill_value=0)
    reduced_df.reset_index(inplace=True)
    grouped_by_zone = aggregate_by_zoning(intersected_zoning, reduced_df, "ID_ZONA_LOC", ["Mujer", "Hombre"], zoning_unique_id)
    return grouped_by_zone


def age_by_zoning(intersected_zoning, zoning_unique_id, path=None):
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
    path: string, default=None
        Ubicación del archivo "Personas.parque" o "Personas.csv" que contienen
        las respuestas del censo.
    
    Returns
    -------
    pandas.dataframe
        Dataframe que contiene el estimado de población correspondiente a cada grupo etario por zona.
    """
    # TODO: incluir parámetro de binning
    region_list = intersected_zoning.REGION.unique()
    age_by_region = loading.read_personas(columnas=['ID_ZONA_LOC', 'P09'], filtros=[("REGION", "in", region_list)], path=path)
    bins = [0, 18, 65, 100, 101]
    labels = ["Menor de edad","Mayor de edad","Adulto mayor", "100 años y más"]
    age_by_region['age_group'] = pd.cut(age_by_region['P09'], bins=bins, labels=labels, include_lowest=True, right=False)
    reduced_df = age_by_region.groupby(["ID_ZONA_LOC"])['age_group'].value_counts().unstack(fill_value=0)
    reduced_df.reset_index(inplace=True)
    grouped_by_zone = aggregate_by_zoning(intersected_zoning, reduced_df, "ID_ZONA_LOC", labels, zoning_unique_id)

    return grouped_by_zone


def inmigrants_by_zoning(intersected_zoning, zoning_unique_id, path=None):
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
    path: string, default=None
        Ubicación del archivo "Personas.parque" o "Personas.csv" que contienen
        las respuestas del censo.
    
    Returns
    -------
    pandas.dataframe
        Dataframe que contiene el estimado de población migrante por zona.
    """
    # TODO: incluir zonas con 0
    region_list = intersected_zoning.REGION.unique()
    # get personal_data
    inmigrants_by_region = loading.read_personas(columnas=['ID_ZONA_LOC', 'P10','P12'], filtros=[("REGION", "in", region_list)], path=path)
    condition_birth = ~inmigrants_by_region['P12'].isin(['Missing', "En esta comuna", "En otra comuna"])
    condition_residence = inmigrants_by_region['P10'] != 'En otro país'
    inmigrants_by_region = inmigrants_by_region[condition_birth & condition_residence]
    agg_count = pd.NamedAgg(column='P12', aggfunc="count")
    reduced_df = inmigrants_by_region.groupby("ID_ZONA_LOC").agg(inmigrantes=agg_count)
    grouped_by_zone = aggregate_by_zoning(intersected_zoning, reduced_df, "ID_ZONA_LOC", ['inmigrantes'], zoning_unique_id)
    
    return grouped_by_zone


def schooling_by_zoning(intersected_zoning, zoning_unique_id, min_age=0, bins=5, path=None):
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
    min_age: int, default=0
        La edad mínina de la población a considerar al calcular las estadísticas.
    bins: int, default=5
        Indica en qué percentil dividir los años de escolaridad. Por defecto se calculan quintiles.
        Si es que la función qcut (TODO: poner link) no logra hacer una segmentación adecuada con el número
        entregado, cambiará la cantidad de bins, por ejemplo de 5 a 4.
    path: string, default=None
        Ubicación del archivo "Personas.parque" o "Personas.csv" que contienen
        las respuestas del censo.
    
    Returns
    -------
    pandas.dataframe
        Dataframe que contiene el estimado de población y años de escolaridad por cada zona.
    """
    region_list = intersected_zoning.REGION.unique()
    schooling_by_region = loading.read_personas(columnas=['ID_ZONA_LOC', 'ESCOLARIDAD'], filtros=[("REGION", "in", region_list), ("ESCOLARIDAD", "<", 30), ("P09", ">=", min_age)], path=path)
    agg_mean = pd.NamedAgg(column='ESCOLARIDAD', aggfunc="mean")
    reduced_df = schooling_by_region.groupby("ID_ZONA_LOC").agg(promedio_escolaridad=agg_mean)
    reduced_df.reset_index(inplace=True)
    groups = pd.qcut(schooling_by_region.ESCOLARIDAD, q=bins, labels=False, retbins=True, duplicates='drop')
    schooling_by_region['schooling_group'] = groups[0].values
    schooling_by_region = schooling_by_region.pivot_table(index='ID_ZONA_LOC', columns='schooling_group', aggfunc='size', fill_value=0)
    schooling_by_region.reset_index(inplace=True)
    new_names = [f'{groups[1][i]:.0f}-{groups[1][i+1]:.0f}' for i in range(len(groups[1])-1)]
    schooling_by_region.rename(columns=dict(map(lambda i,j : (i,j) , range(len(new_names)),new_names)), inplace=True)
    reduced_df = reduced_df.merge(schooling_by_region, on='ID_ZONA_LOC')
    census_data = intersected_zoning.merge(reduced_df, on="ID_ZONA_LOC")
    census_data["promedio_pond"] = census_data['percentage_overlap']*census_data['promedio_escolaridad']
    for label in new_names:
        census_data[label] = census_data[label] * census_data['percentage_overlap']
    new_names += ['percentage_overlap', 'promedio_pond']
    grouped_by_zone = census_data.groupby(zoning_unique_id)[new_names].sum()
    grouped_by_zone['promedio'] = grouped_by_zone["promedio_pond"]/grouped_by_zone["percentage_overlap"]
    grouped_by_zone = grouped_by_zone.drop(['percentage_overlap', 'promedio_pond'], axis=1)
    grouped_by_zone.reset_index(inplace=True)
    return grouped_by_zone


def indigenous_by_zoning(intersected_zoning, zoning_unique_id, path=None):
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
    path: string, default=None
        Ubicación del archivo "Personas.parque" o "Personas.csv" que contienen
        las respuestas del censo.
    
    Returns
    -------
    pandas.dataframe
        Dataframe que contiene el estimado de población por zona.
    """
    region_list = intersected_zoning.REGION.unique()
    # get personal_data
    df = loading.read_personas(columnas=['ID_ZONA_LOC', 'P16'], filtros=[("REGION", "in", region_list)], path=path)
    df = df[df['P16']=='Sí']
    agg_count = pd.NamedAgg(column='P16', aggfunc="count")
    reduced_df = df.groupby("ID_ZONA_LOC").agg(pueblo_originario=agg_count)
    reduced_df.reset_index(inplace=True)
    grouped_by_zone = aggregate_by_zoning(intersected_zoning, reduced_df, "ID_ZONA_LOC", ['pueblo_originario'], zoning_unique_id)
    
    return grouped_by_zone

