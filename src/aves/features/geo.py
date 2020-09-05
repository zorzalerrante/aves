import shapely
import geopandas as gpd
import numpy as np
import KDEpy

def clip_area_geodataframe(geodf, bounding_box):
    # bounding box should be in same crs
    
    bounds = shapely.geometry.box(*bounding_box)
    
    try:
        return (geodf
                .assign(geometry=lambda g: g.map(lambda x: x.intersection(bounds)))
                .pipe(lambda x: x[x.geometry.area > 0])
        )
    except Exception:
        # sometimes there are ill-defined intersections in polygons.
        return (geodf
            .assign(geometry=lambda g: g.buffer(0).map(lambda x: x.intersection(bounds)))
            .pipe(lambda x: x[x.geometry.area > 0])
        )

    
def to_point_geodataframe(df, longitude, latitude, crs='epsg:4326', drop=False):
    gdf = gpd.GeoDataFrame(df, geometry=df.apply(lambda x: shapely.geometry.Point((x[longitude], x[latitude])), axis=1), crs=crs)
    
    if drop:
        gdf = gdf.drop([longitude, latitude], axis=1)
    
    return gdf


def kde_from_points(geodf, kernel='gaussian', norm=2, bandwidth=1e-2, grid_points=2**9, weight_column=None):
    # La variable grid_points define la cantidad de puntos en el espacio en el que se estimará la densidad
    # hacemos una lista con las coordenadas de los viajes
    point_coords = np.vstack([geodf.geometry.x, geodf.geometry.y]).T
    # instanciamos la Fast-Fourier Transform Kernel Density Estimation
    kde = KDEpy.FFTKDE(bw=bandwidth, norm=norm, kernel=kernel)
    weights = None if weight_column is None else geodf[weight_column].values
    grid, points = kde.fit(point_coords, weights=weights).evaluate(grid_points)
    x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    z = points.reshape(grid_points, grid_points).T
    return x, y, z