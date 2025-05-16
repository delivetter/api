from fastapi import APIRouter, FastAPI, Security
from fastapi.responses import HTMLResponse
from lib.barrios import get_barrio_polygon
from lib.geojson import geojson_to_graph
from lib.map import map_to_html

from . import __project_name__
from .auth import JWTBearer
from .lib.models import simulation_M1, simulation_M2, generate_context
from .models import Input, ModelOutput, SimulateOutput

app = FastAPI(title=__project_name__, allow_origins=["*"])

public_router = APIRouter()
secured_router = APIRouter(dependencies=[Security(JWTBearer())])


# @public_router.get("/simulate/m1/map", response_class=HTMLResponse)
# def map_m1():
#     input = Input(
#         barrio="BENIMACLET",
#         nodo_entrada=257843726,
#         almacen="almacen_24814",
#         num_paquetes=30,
#     )
#     G_barrio = geojson_to_graph(f"src/data/supergrafos/{input.barrio}.geojson")
#     df_comercios = generate_context(G_barrio, input.num_paquetes)
#     barrio_polygon = get_barrio_polygon(input.barrio)

#     _, mapa_m1 = simulation_M1(
#         G_barrio,
#         df_comercios,
#         input.nodo_entrada,
#         shp_zone=barrio_polygon,
#     )

#     return map_to_html(mapa_m1)


@secured_router.post("/simulate/m1/map", response_class=HTMLResponse)
def map_m1(input: Input):
    G_barrio = geojson_to_graph(f"src/data/supergrafos/{input.barrio}.geojson")
    df_comercios = generate_context(G_barrio, input.num_paquetes)
    barrio_polygon = get_barrio_polygon(input.barrio)

    _, mapa_m1 = simulation_M1(
        G_barrio,
        df_comercios,
        input.nodo_entrada,
        shp_zone=barrio_polygon,
    )

    return map_to_html(mapa_m1)


@secured_router.post("/simulate/m2/map", response_class=HTMLResponse)
def map_m2(input: Input):
    G_barrio = geojson_to_graph(f"src/data/supergrafos/{input.barrio}.geojson")
    df_comercios = generate_context(G_barrio, input.num_paquetes)
    barrio_polygon = get_barrio_polygon(input.barrio)

    _, mapa_m2 = simulation_M2(
        G_barrio,
        df_comercios,
        input.nodo_entrada,
        input.almacen,
        shp_zone=barrio_polygon,
    )

    return map_to_html(mapa_m2)


@secured_router.post("/simulate/m1")
def simulate_m1(input: Input) -> ModelOutput:
    G_barrio = geojson_to_graph(f"src/data/supergrafos/{input.barrio}.geojson")
    df_comercios = generate_context(G_barrio, input.num_paquetes)
    barrio_polygon = get_barrio_polygon(input.barrio)

    results_m1, mapa_m1 = simulation_M1(
        G_barrio,
        df_comercios,
        input.nodo_entrada,
        shp_zone=barrio_polygon,
    )

    return {"results": results_m1, "map_html": map_to_html(mapa_m1)}


@secured_router.post("/simulate/m2")
def simulate_m2(input: Input) -> ModelOutput:
    G_barrio = geojson_to_graph(f"src/data/supergrafos/{input.barrio}.geojson")
    df_comercios = generate_context(G_barrio, input.num_paquetes)
    barrio_polygon = get_barrio_polygon(input.barrio)

    results_m2, mapa_m2 = simulation_M2(
        G_barrio,
        df_comercios,
        input.nodo_entrada,
        input.almacen,
        shp_zone=barrio_polygon,
    )

    return {"results": results_m2, "map_html": map_to_html(mapa_m2)}


@secured_router.post("/simulate")
def simulate(input: Input) -> SimulateOutput:
    G_barrio = geojson_to_graph(f"src/data/supergrafos/{input.barrio}.geojson")
    df_comercios = generate_context(G_barrio, input.num_paquetes)
    barrio_polygon = get_barrio_polygon(input.barrio)

    results_m1, mapa_m1 = simulation_M1(
        G_barrio,
        df_comercios,
        input.nodo_entrada,
        shp_zone=barrio_polygon,
    )
    results_m2, mapa_m2 = simulation_M2(
        G_barrio,
        df_comercios,
        input.nodo_entrada,
        input.almacen,
        shp_zone=barrio_polygon,
    )

    return {
        "m1": {"results": results_m1, "map_html": map_to_html(mapa_m1)},
        "m2": {"results": results_m2, "map_html": map_to_html(mapa_m2)},
    }


app.include_router(public_router, tags=["public"])
app.include_router(secured_router, tags=["secured"])
