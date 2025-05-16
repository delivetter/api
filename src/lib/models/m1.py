import math
from typing import TypedDict
from lib.barrio_graph import get_nodos_carga
from lib.models.common.calculate_distances import haversine_dist
from lib.models.common.types import ModelResults
import pandas as pd
import numpy as np
import networkx as nx
import geopandas as gpd
from shapely import Polygon
from shapely.geometry import Point
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium
from folium import FeatureGroup, DivIcon
from .common.calculate_time import calcular_tiempo_walk, calcular_tiempo_drive

FIXED_DAILY_RATE_VAN = 31.76  # €/día
TIME_HOURLY_RATE_VAN = 21.19 + FIXED_DAILY_RATE_VAN / 8  # €/hora
DISTANCE_PER_KM_RATE_VAN = 0.184  # €/km


class CIDRoute(TypedDict):
    deposito: int
    sequence: list[int]
    time_min: float
    distance_km: int


def cid_assigment(
    G_super: nx.DiGraph,
    df_comercios: pd.DataFrame,
    nodos_carga: list[int],
    capacidad_maxima: int,
    extra_penalty=0.1,
):
    """
    Heurística Marginal Trips.

    Args:
    df_comercios     : GeoDataFrame con ['node','paquetes','geometry'].
    G_super          : Grafo de conducción.
    nodos_carga      : lista de nodos CID.
    CAPACIDAD_MAXIMA : capacidad máxima de carga del repartidor.
    extra_penalty    : factor multiplicativo para penalizar viajes extra.

    Devuelve:
    assign_df : DataFrame con ['commerce_id','paquetes','assigned_cid',
                                'walk_time_min','delta_runs'].
    loads_df  : GeoDataFrame con ['node','geometry','total_load','required_runs'].
    """
    # Construir df_cid
    df_cid = gpd.GeoDataFrame(
        {"node": nodos_carga},
        geometry=[Point(G_super.nodes[n]["x"], G_super.nodes[n]["y"]) for n in nodos_carga],
        crs=df_comercios.crs,
    )

    # 1) Matriz de tiempos a pie
    n, m = len(df_comercios), len(df_cid)
    time_walk = np.zeros((n, m))
    for i, ci in df_comercios.iterrows():
        for j, sj in df_cid.iterrows():
            t, _ = calcular_tiempo_walk(G_super, sj["node"], ci["node"])
            time_walk[i, j] = t

    # 2) Asignación Marginal Trips
    L = np.zeros(m, dtype=int)
    assignments = []
    for i, ci in df_comercios.iterrows():
        best_cost, best_j = float("inf"), None
        for j, sj in df_cid.iterrows():
            runs_before = math.ceil(L[j] / capacidad_maxima)
            runs_after = math.ceil((L[j] + ci["paquetes"]) / capacidad_maxima)
            delta_runs = runs_after - runs_before
            cost = time_walk[i, j] + time_walk[i, j] * delta_runs * extra_penalty
            if cost < best_cost:
                delta_f = delta_runs
                best_cost, best_j = cost, j

        assignments.append(
            {
                "commerce_id": ci["node"],
                "paquetes": ci["paquetes"],
                "assigned_cid": df_cid.loc[best_j, "node"],
                "walk_time_min": time_walk[i, best_j],
                "delta_runs": delta_f,
            }
        )
        L[best_j] += ci["paquetes"]

    assign_df = pd.DataFrame(assignments)

    # 3) Cargas y viajes requeridos
    loads_df = gpd.GeoDataFrame(
        {
            "node": df_cid["node"],
            "geometry": df_cid["geometry"],
            "total_load": L,
            "required_runs": np.ceil(L / capacidad_maxima).astype(int),
        },
        crs=df_cid.crs,
    )

    return assign_df, loads_df


def walk_routes(
    G_super: nx.DiGraph, assign_df: pd.DataFrame, loads_df: pd.DataFrame, capacidad_maxima: int
) -> pd.DataFrame:
    """
    CVRP con OR-Tools, tiempos en segundos, manejo de 'inf'
    y cómputo de distancia recorrida por ruta.

    Args:
    G_super            : Grafo de calles.
    assign_df          : DataFrame con ['commerce_id','paquetes','assigned_cid'].
    loads_df           : GeoDataFrame con ['node',...].
    capacidad_maxima   : int, Q paquetes por ruta.

    Returns:
    routes_df : DataFrame ['cid','batch','sequence','time_min','distance_km','load'].
    """
    all_routes = []
    total_time_sec = 0
    total_batches = 0
    INF_SEC = 10**9

    for _, row in loads_df.iterrows():
        cid = row["node"]
        assigned = assign_df[assign_df["assigned_cid"] == cid]
        if assigned.empty:
            continue

        # 1) nodos y demandas
        locations = [cid] + assigned["commerce_id"].tolist()
        demands = [0] + assigned["paquetes"].tolist()
        N = len(locations)

        # 2) matriz de tiempos (segundos)
        time_matrix = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                try:
                    t_min, _ = calcular_tiempo_walk(
                        G_super, locations[i], locations[j]
                    )  # Intentamos con G_super
                except Exception:
                    t_min = INF_SEC
                time_matrix[i][j] = (
                    INF_SEC if not np.isfinite(t_min) else int(math.ceil(t_min * 60))
                )

        # 3) número de rutas necesarias
        total_demand = sum(demands)
        R = math.ceil(total_demand / capacidad_maxima)

        # 4) configurar OR-Tools
        manager = pywrapcp.RoutingIndexManager(N, R, 0)
        routing = pywrapcp.RoutingModel(manager)

        def time_callback(f, t):
            return time_matrix[manager.IndexToNode(f)][manager.IndexToNode(t)]

        cb_t = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(cb_t)

        def demand_callback(idx):
            return demands[manager.IndexToNode(idx)]

        cb_d = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            cb_d, 0, [capacidad_maxima + 1] * R, True, "Capacity"
        )

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS
        search_params.time_limit.seconds = 3

        try:
            sol = routing.SolveWithParameters(search_params)
            if not sol:
                raise RuntimeError(f"No solution CVRP for CID {cid}")
        except RuntimeError:
            (f"No solution CVRP for CID {cid}")
            continue
        # 5) extraer rutas y calcular distancia
        for v in range(R):
            idx = routing.Start(v)
            if routing.IsEnd(sol.Value(routing.NextVar(idx))):
                continue
            seq_nodes = []
            load = 0
            time_sec = 0
            dist_m = 0.0
            # construir ruta
            while not routing.IsEnd(idx):
                n = manager.IndexToNode(idx)
                seq_nodes.append(locations[n])
                load += demands[n]
                nxt = sol.Value(routing.NextVar(idx))
                time_sec += routing.GetArcCostForVehicle(idx, nxt, v)

                # distancia tramo a tramo
                a, b = locations[n], locations[manager.IndexToNode(nxt)]
                _, path = calcular_tiempo_walk(G_super, a, b)  # Usamos G_super para la distancia
                if path:
                    for u, w in zip(path[:-1], path[1:]):
                        ed = G_super.get_edge_data(u, w)
                        dist_m += min(e.get("length", 0) for e in ed.values())

                idx = nxt

            # retorno opcional al depósito
            seq_nodes.append(cid)

            all_routes.append(
                {
                    "cid": cid,
                    "travel_number": v + 1,
                    "sequence": seq_nodes,
                    "time_min": round(time_sec / 60, 3),
                    "distance_km": round(dist_m / 1000, 3),
                    "load": load,
                }
            )
            total_time_sec += time_sec
            total_batches += 1

    routes_df = pd.DataFrame(all_routes)
    return routes_df


def drive_route(G_super: nx.DiGraph, routes_df: pd.DataFrame, nodo_entrada: int) -> CIDRoute:
    """
    Calcula el recorrido óptimo que pasa por todos los CIDs únicos en routes_df,
    utilizando el grafo de conducción G_super y la función calcular_tiempo_drive.

    Args:
    G_super      : Grafo de conducción.
    routes_df    : DataFrame con ['cid', 'travel_number', 'sequence', ...] que contiene las rutas.
    nodo_entrada : nodo por donde llega la furgoneta
    Returns:
    cid_route    : DataFrame con las rutas entre los CIDs, tiempos y distancias calculados.
    """

    unique_cids = routes_df["cid"].unique()
    N = len(unique_cids) + 1  # +1 porque también contamos con el nodo de depósito
    time_matrix = [[0] * N for _ in range(N)]

    # Llenamos la matriz de tiempos entre todos los CIDs y el depósito
    cid_to_index = {cid: idx + 1 for idx, cid in enumerate(unique_cids)}
    cid_to_index["deposit"] = 0  # El nodo del depósito estará en la posición 0

    # Rellenar la matriz con los tiempos de conducción
    for i, cid1 in enumerate([nodo_entrada] + list(unique_cids)):
        for j, cid2 in enumerate([nodo_entrada] + list(unique_cids)):
            if i != j:
                # Calculamos el tiempo de conducción entre los CIDs
                try:
                    t_min, _ = calcular_tiempo_drive(G_super, cid1, cid2)
                except Exception:
                    t_min = float("inf")  # Si hay un error, asignamos un valor infinito
                if np.isfinite(t_min):
                    time_matrix[i][j] = int(
                        math.ceil(t_min * 60)
                    )  # Convertimos minutos a segundos

    # 4) Resolver el TSP con OR-Tools
    manager = pywrapcp.RoutingIndexManager(
        N, 1, 0
    )  # 1 vehículo, comenzamos desde el nodo 0 (depósito)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        # Llamada a la función para obtener el tiempo entre dos nodos
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Parámetros de búsqueda
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS
    search_parameters.time_limit.seconds = 3  # Limitar el tiempo de búsqueda

    # Resolvemos el problema de TSP
    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        raise RuntimeError("No se pudo encontrar una solución para el TSP")

    # 5) Extraer la ruta óptima y calcular la distancia total
    route_sequence = [nodo_entrada]  # Iniciamos con el nodo de depósito
    total_time_sec = 0
    total_distance_m = 0
    idx = routing.Start(0)

    while not routing.IsEnd(idx):
        node_index = manager.IndexToNode(idx)
        cid = list(unique_cids)[node_index - 1]  # Obtener el CID correspondiente
        route_sequence.append(cid)  # Agregar CID correspondiente
        next_idx = solution.Value(routing.NextVar(idx))
        total_time_sec += routing.GetArcCostForVehicle(idx, next_idx, 0)

        # Calculamos la distancia entre los nodos
        a, b = route_sequence[-2], route_sequence[-1]  # Usamos los últimos dos CIDs
        _, path = calcular_tiempo_drive(G_super, a, b)
        if path:
            for u, w in zip(path[:-1], path[1:]):
                ed = G_super.get_edge_data(u, w)
                total_distance_m += min(e.get("length", 0) for e in ed.values())

        idx = next_idx

    # Añadimos el nodo de vuelta al inicio para cerrar la ruta

    total_time_min = round(total_time_sec / 60, 3)
    total_distance_km = round(total_distance_m / 1000, 3)

    # 6) Crear el DataFrame con la ruta calculada
    cid_route: CIDRoute = {
        "deposito": nodo_entrada,
        "sequence": route_sequence,  # Aquí se crea la lista de la columna 'cid'
        "time_min": total_time_min,
        "distance_km": total_distance_km,
    }

    return cid_route


def calculate_results_m1(routes_df: pd.DataFrame, cid_route: CIDRoute) -> ModelResults:
    """
    Calcula el coste total del reparto sumando costes fijos, por tiempo y por distancia.

    Args:
        routes_df    : rutas andando
        cid_route    : ruta furgone

    Returns:
        total_cost   : Diccionario con los costes calculados.
    """
    total_kms_walk = routes_df["distance_km"].sum()
    total_hours_walk = routes_df["time_min"].sum() / 60
    total_kms_drive = cid_route["distance_km"]
    total_hours_drive = cid_route["time_min"] / 60

    distance_cost_van = total_kms_drive * DISTANCE_PER_KM_RATE_VAN
    distance_cost_ona = 0
    time_cost_van = (total_hours_drive + total_hours_walk) * TIME_HOURLY_RATE_VAN
    time_cost_ona = 0

    results = {
        "total_kms_walk": total_kms_walk,
        "total_hours_walk": total_hours_walk,
        "total_kms_drive": total_kms_drive,
        "total_hours_drive": total_hours_drive,
        "distance_cost_van": distance_cost_van,
        "distance_cost_ona": distance_cost_ona,
        "time_cost_van": time_cost_van,
        "time_cost_ona": time_cost_ona,
        "total_cost": distance_cost_van + time_cost_van + distance_cost_ona + time_cost_ona,
    }

    return {categoria: round(valor, 4) for categoria, valor in results.items()}


def map_m1(
    G_super: nx.DiGraph,
    assign_df: pd.DataFrame,
    df_comercios: gpd.GeoDataFrame,
    routes_df: pd.DataFrame,
    cid_route: CIDRoute,
    capacidad_maxima: int,
    min_dist_m=50,
    shp_zone=None,
) -> folium.Map:
    """
    Dibuja un mapa interactivo combinando el resultado de las asignaciones de CID,
    las rutas a pie, y las rutas de furgoneta.

    Args:
    G_super        : Grafo de las calles, usado para obtener las coordenadas de los nodos.
    assign_df     : DataFrame con ['commerce_id','paquetes','assigned_cid',...].
    df_comercios  : GeoDataFrame con ['node','paquetes','geometry'].
    routes_df      : DataFrame con las rutas generadas, con columnas ['cid', 'sequence', 'time_min', 'distance_km', 'load'].
    cid_route      : DataFrame con la ruta de la furgoneta, con columnas ['deposito', 'sequence', 'time_min', 'num_furgonetas'].
    shp_zone       : Opción de polígono de zona (opcional).

    Devuelve:
    folium.Map con:
    - Grafo de calles.
    - Comercios coloreados según CID asignado.
    - Rutas a pie y de furgoneta, con flechas y marcadores.
    - Control de capas para activar/desactivar las rutas.
    """
    # 1) Preparar paleta de colores para CID
    valid_colors = [
        "blue",
        "green",
        "red",
        "purple",
        "orange",
        "darkred",
        "cadetblue",
        "darkblue",
        "darkgreen",
        "lightgreen",
        "pink",
        "gray",
    ]
    unique_cids = sorted(assign_df["assigned_cid"].unique().tolist())
    cid_color_map = {cid: valid_colors[i % len(valid_colors)] for i, cid in enumerate(unique_cids)}

    # 2) Mapa base
    centroide = df_comercios.geometry.unary_union.centroid
    centro = [centroide.y, centroide.x]
    m = folium.Map(location=centro, zoom_start=16, control_scale=True, tiles="cartodb positron")

    # 3) Polígono de zona (si corresponde)
    if shp_zone is not None:
        folium.GeoJson(
            shp_zone,
            style_function=lambda f: {
                "color": "green",
                "weight": 2,
                "fill": True,
                "fillColor": "limegreen",
                "fillOpacity": 0.1,
            },
            control=False,
        ).add_to(m)

    # 4) Grafo de calles en gris
    for u, v, _ in G_super.edges(data=True):
        y0, x0 = G_super.nodes[u]["y"], G_super.nodes[u]["x"]
        y1, x1 = G_super.nodes[v]["y"], G_super.nodes[v]["x"]
        folium.PolyLine([(y0, x0), (y1, x1)], color="lightgray", weight=1, opacity=0.3).add_to(m)

    # 5) Comercios coloreados por CID asignado
    df_merged = df_comercios.merge(
        assign_df[["commerce_id", "assigned_cid"]], left_on="node", right_on="commerce_id"
    )
    for _, row in df_merged.iterrows():
        color = cid_color_map[row["assigned_cid"]]
        folium.CircleMarker(
            location=(row.geometry.y, row.geometry.x),
            radius=6,
            color="black",
            weight=0.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=f"{row['node']} \nPaquetes: {row['paquetes']}",
        ).add_to(m)

    # 6) Hubs de CID (marcadores de camión)
    hubs_layer = FeatureGroup(name="CID assigments", show=True)
    loads = assign_df.groupby("assigned_cid")["paquetes"].sum().reset_index()
    loads["required_runs"] = np.ceil(loads["paquetes"] / capacidad_maxima).astype(int)
    loads["geometry"] = loads["assigned_cid"].apply(
        lambda cid: Point(G_super.nodes[cid]["x"], G_super.nodes[cid]["y"])
    )
    for _, r in loads.iterrows():
        cid = r["assigned_cid"]
        color = cid_color_map[cid]
        y, x = r["geometry"].y, r["geometry"].x
        folium.Marker(
            location=(y, x),
            icon=folium.Icon(prefix="fa", icon="truck", color=color),
            popup=f"{cid} ({r['required_runs']} viajes)",
        ).add_to(hubs_layer)
    hubs_layer.add_to(m)

    # 7) Rutas a pie con flechas
    route_layer = FeatureGroup(name="CID Routes", show=True)
    for _, row in routes_df.iterrows():
        cid = row["cid"]
        color = cid_color_map[cid]
        seq = row["sequence"]
        coords = []
        for a, b in zip(seq[:-1], seq[1:]):
            _, path = calcular_tiempo_walk(G_super, a, b)
            if path:
                coords.extend([(G_super.nodes[n]["y"], G_super.nodes[n]["x"]) for n in path])

        if len(coords) < 2:
            continue

        # Dibujar PolyLine punteada
        folium.PolyLine(
            coords, color=color, weight=3, opacity=0.5, dash_array="5,10", popup=f"CID {cid}"
        ).add_to(route_layer)

        # Calcular midpoints + ángulos para las flechas
        mids = []
        for (lat1, lon1), (lat2, lon2) in zip(coords[:-1], coords[1:]):
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2
            dx, dy = lon2 - lon1, lat2 - lat1
            angle = math.degrees(math.atan2(dy, dx))
            mids.append({"pt": (mid_lat, mid_lon), "angle": -angle})

        # Filtrar por distancia mínima
        filtered = []
        last_pt = None
        for item in mids:
            if last_pt is None or haversine_dist(last_pt, item["pt"]) >= min_dist_m:
                filtered.append(item)
                last_pt = item["pt"]

        # Colocar flechas en los midpoints filtrados
        for itm in filtered:
            lat, lon = itm["pt"]
            html = f"""
            <div style="
                transform: rotate({itm['angle']}deg);
                font-size:12px;
                color:{color};
                opacity:0.9;
            ">➤</div>"""
            folium.Marker(
                location=(lat, lon),
                icon=DivIcon(icon_size=(20, 20), icon_anchor=(10, 10), html=html),
            ).add_to(route_layer)

    route_layer.add_to(m)

    # 8) Ruta de furgoneta (truck route)
    truck_layer = FeatureGroup(name="Truck Route", show=True)
    seq = cid_route["sequence"]
    coords = []
    for a, b in zip(seq[:-1], seq[1:]):
        _, path = calcular_tiempo_drive(G_super, a, b)
        coords += [(G_super.nodes[n]["y"], G_super.nodes[n]["x"]) for n in path]
        color = cid_color_map[b]
        y, x = G_super.nodes[b]["y"], G_super.nodes[b]["x"]
        folium.Marker(
            location=(y, x),
            icon=folium.Icon(prefix="fa", icon="truck", color=color),
            popup=f"{b} ({loads.loc[loads['assigned_cid'] == b, 'required_runs'].values[0]} viajes)",
        ).add_to(truck_layer)
    
    folium.PolyLine(
        coords,
        color='black',
        weight=6,
        opacity=0.3,
        tooltip=f"Truck route: {cid_route['time_min']} min"
    ).add_to(truck_layer)
    
    # 9) Marcador del depósito
    dep = cid_route["deposito"]
    y_dep = G_super.nodes[dep]["y"]
    x_dep = G_super.nodes[dep]["x"]
    folium.Marker(
        location=(y_dep, x_dep),
        icon=folium.Icon(prefix="fa", icon="play", color="black"),
        popup=f"Depósito: {dep}",
    ).add_to(truck_layer)

    truck_layer.add_to(m)

    # 10) Control de capas
    folium.LayerControl(collapsed=False).add_to(m)

    return m


def simulation_M1(
    G_super: nx.DiGraph,
    df_comercios: pd.DataFrame,
    nodo_entrada: int,
    shp_zone: Polygon,
    capacidad_maxima=8,
) -> tuple[ModelResults, folium.Map]:
    nodos_carga = get_nodos_carga(G_super)
    assign_df, loads_df = cid_assigment(
        G_super, df_comercios, nodos_carga, capacidad_maxima, extra_penalty=0.05
    )
    routes_df = walk_routes(G_super, assign_df, loads_df, capacidad_maxima)
    cid_route = drive_route(G_super, routes_df, nodo_entrada)

    map = map_m1(
        G_super, assign_df, df_comercios, routes_df, cid_route, capacidad_maxima, shp_zone=shp_zone
    )
    results = calculate_results_m1(routes_df, cid_route)

    return results, map
