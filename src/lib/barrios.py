from shapely import Polygon
import geopandas as gpd


BARRIOS = gpd.read_file("src/data/barris.geojson")


def get_barrio_polygon(barrio: str) -> Polygon:
    return BARRIOS[BARRIOS["nombre"] == barrio].geometry.values[0]
