import s2sphere
from io import BytesIO

try:
    import rapidjson as json
except ImportError:
    import json

import geopandas as gpd

from .base import Grid

class S2Grid(Grid):
    @classmethod
    def __init__(self, bounds, grid_level=12, extra_margin=0.0, crs="epsg:4326"):
        bounds = list(bounds)
        bounds[0] = bounds[0] - extra_margin * (bounds[2] - bounds[0])
        bounds[2] = bounds[2] + extra_margin * (bounds[2] - bounds[0])
        bounds[1] = bounds[1] - extra_margin * (bounds[3] - bounds[1])
        bounds[3] = bounds[3] + extra_margin * (bounds[3] - bounds[1])

        r = s2sphere.RegionCoverer()
        r.min_level = grid_level
        r.max_level = grid_level
        p1 = s2sphere.LatLng.from_degrees(bounds[1], bounds[0])
        p2 = s2sphere.LatLng.from_degrees(bounds[3], bounds[2])

        cell_ids = r.get_covering(s2sphere.LatLngRect.from_point_pair(p1, p2))

        def s2geojson(cellids):
            fc = {"type": "FeatureCollection"}

            features = []

            for cid in cellids:
                pt_feature = {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {"type": "Polygon"},
                }
                cell = s2sphere.Cell(cid)
                vertices = [
                    s2sphere.LatLng.from_point(cell.get_vertex(v)) for v in range(4)
                ]
                vertices = [[v.lng().degrees, v.lat().degrees] for v in vertices]
                pt_feature["geometry"]["coordinates"] = [
                    vertices
                ]  # [float(s) for s in str(latlong).split()[-1].split(',')
                pt_feature["properties"]["s2_cellid"] = str(cid.id())

                features.append(pt_feature)

            fc["features"] = features

            return fc

        grid = s2geojson(cell_ids)
        self.geodf = gpd.read_file(BytesIO(json.dumps(grid).encode("utf-8")), driver="GeoJSON")
        self.geodf.crs = crs
        self.zoom_level = grid_level
