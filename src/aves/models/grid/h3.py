import h3

try:
    import rapidjson as json
except ImportError:
    import json

import geopandas as gpd
from shapely.geometry import Polygon

from .base import Grid


class MockGeo:
    def __init__(self, d):
        self.d = d

    @property
    def __geo_interface__(self):
        return self.d


class H3Grid(Grid):
    def __init__(self, bounds, extra_margin=0.0, grid_level=12, crs="epsg:4326"):
        bounds = list(bounds)
        bounds[0] = bounds[0] - extra_margin * (bounds[2] - bounds[0])
        bounds[2] = bounds[2] + extra_margin * (bounds[2] - bounds[0])
        bounds[1] = bounds[1] - extra_margin * (bounds[3] - bounds[1])
        bounds[3] = bounds[3] + extra_margin * (bounds[3] - bounds[1])

        cell_ids = h3.h3shape_to_cells(
            h3.geo_to_h3shape(
                MockGeo(
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [bounds[0], bounds[1]],
                                [bounds[2], bounds[1]],
                                [bounds[2], bounds[3]],
                                [bounds[0], bounds[3]],
                            ]
                        ],
                    }
                )
            ),
            res=grid_level,
        )

        h3grid = gpd.GeoDataFrame(
            {"h3_cell_id": list(map(str, cell_ids))},
            geometry=[
                Polygon(
                    list(
                        map(lambda x: tuple(reversed(x)), h3.cell_to_boundary(cell_id))
                    )
                )
                for cell_id in cell_ids
            ],
            crs="epsg:4326",
        ).to_crs(crs)

        self.geodf = h3grid
        self.zoom_level = grid_level


# def h3_to_poly(code):
#     coords = h3.cell_to_boundary(code)
#     coords = map(lambda x: list(reversed(x)), coords)
#     return list(coords)
