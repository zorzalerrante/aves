
from abc import ABC, abstractmethod

class Grid(ABC):
    def __init__(self, bounds, grid_level, extra_margin, crs):
        pass

    @classmethod 
    def from_geodf(cls, geodf, grid_level=12, extra_margin=0.0):
        return cls(geodf.to_crs('epsg:4326').total_bounds, grid_level=grid_level, extra_margin=extra_margin, crs=geodf.crs)