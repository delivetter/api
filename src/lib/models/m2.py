import math
import random
from typing import Iterable, TypedDict
from lib.models.common.calculate_distances import haversine_dist
from lib.models.common.types import ModelResults
import pandas as pd
import numpy as np
import networkx as nx
import geopandas as gpd
from shapely import Polygon
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import distance_metric, type_metric
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler
from .common.calculate_time import calcular_tiempo_walk, calcular_tiempo_drive
import matplotlib.colors as mcolors

FIXED_DAILY_RATE_VAN = 31.76  # €/día
TIME_HOURLY_RATE_VAN = 21.19 + FIXED_DAILY_RATE_VAN / 8  # €/hora
DISTANCE_PER_KM_RATE_VAN = 0.184  # €/km

FIXED_DAILY_RATE_ONA = 35.42  # €/día
TIME_HOURLY_RATE_ONA = 9.58 + FIXED_DAILY_RATE_ONA / 8  # €/hora
DISTANCE_PER_KM_RATE_ONA = 0.055  # €/km


class HubRoute(TypedDict):
    depot: int
    hub: str
    t_drive: float
    dist: float
    camino: Iterable


def planificar_hub(G_super: nx.MultiDiGraph, nodo_entrada: int, hub: str) -> HubRoute | None:
    """
    Selecciona un hub de almacén cercano al centroide del grafo y calcula la ruta
    desde un nodo de entrada hasta ese hub. Si el hub elegido falla, reintenta con
    los demás candidatos.

    Parámetros:
    - nodo_entrada: Nodo inicial (depósito periférico) desde el que partimos.
    - nodos_almacenes: Lista de nodos de almacenes.
    - percentil: Percentil de almacenes más cercanos al centroide a considerar (por defecto 50).
    - seed: Semilla para la aleatoriedad (opcional).

    Retorna:
    - dict con:
        'depot'   : nodo_entrada,
        'hub'     : nodo seleccionado con ruta válida,
        't_drive' : tiempo de conducción al hub (en minutos),
        'dist'    : distancia total (en km),
        'camino'  : lista de nodos que forman la ruta
    o None si ningún candidato tiene ruta.
    """
    t_drive, camino = calcular_tiempo_drive(G_super, nodo_entrada, hub)
    if np.isfinite(t_drive):
        distancia_m = sum(
            G_super.get_edge_data(u, v)[0]["length"] for u, v in zip(camino[:-1], camino[1:])
        )
        return {
            "depot": nodo_entrada,
            "hub": hub,
            "t_drive": t_drive,
            "dist": distancia_m / 1000,
            "camino": camino,
        }

    # Si agotamos todos los candidatos sin éxito:
    return None


def clustering(
    df_comercios: gpd.GeoDataFrame,
    cap_max=20,
    k_min=2,
    k_max=10,
    alpha=0.2,
    beta=0.2,
    gamma=0.6,
):
    coords = np.vstack([df_comercios.geometry.x.values, df_comercios.geometry.y.values]).T
    demands = df_comercios["paquetes"].to_numpy()

    ks: list[int] = []
    sils, ines, balances = (
        [],
        [],
        [],
    )

    for k in range(k_min, min(k_max + 1, len(df_comercios))):
        # Inicializar indices aleatorios para medoides
        rng = np.random.default_rng(42)
        initial_medoids = rng.choice(len(coords), size=k, replace=False).tolist()

        # Crear instancia de K-Medoids con distancia Euclidea
        metric = distance_metric(type_metric.EUCLIDEAN)
        kmed = kmedoids(coords.tolist(), initial_medoids, metric=metric)
        kmed.process()

        clusters = kmed.get_clusters()
        medoid_indices = kmed.get_medoids()

        # Etiquetas para cada punto
        labels = np.empty(len(coords), dtype=int)
        for i, cluster in enumerate(clusters):
            labels[cluster] = i

        # Silhouette
        sil = silhouette_score(coords, labels) if k > 1 else 0

        # Inertia
        medoids = coords[medoid_indices]
        inertia = sum(np.sum((coords[labels == j] - medoids[j]) ** 2) for j in range(k))

        # Balance de capacidad
        loads = np.array([demands[labels == j].sum() for j in range(k)])
        mults = np.ceil(loads / cap_max) * cap_max
        errors = (loads - mults) ** 2
        bal = np.mean(errors)

        ks.append(k)
        sils.append(sil)
        ines.append(inertia)
        balances.append(bal)

    # Escalado y combinación
    scaler = RobustScaler()
    sil_r = scaler.fit_transform(np.array(sils).reshape(-1, 1)).ravel()
    ine_r = -scaler.fit_transform(np.array(ines).reshape(-1, 1)).ravel()
    bal_r = -scaler.fit_transform(np.array(balances).reshape(-1, 1)).ravel()
    combined = alpha * sil_r + beta * ine_r + gamma * bal_r

    best_idx = np.argmax(combined)
    best_k: int = ks[best_idx]

    # Reclustering con mejor k
    rng = np.random.default_rng(43)
    initial_medoids = rng.choice(len(coords), size=best_k, replace=False).tolist()
    kmed = kmedoids(coords.tolist(), initial_medoids, metric=metric)
    kmed.process()

    clusters = kmed.get_clusters()
    labels = np.empty(len(coords), dtype=int)
    for i, cluster in enumerate(clusters):
        labels[cluster] = i

    df_comercios["grupo_repartidor"] = labels + 1
    return best_k, df_comercios


def cluster_routes(
    G_super: nx.MultiDiGraph,
    df_comercios: gpd.GeoDataFrame,
    hub_route: HubRoute,
    capacidad_maxima: int,
):
    """
    Calcular rutas de entrega para cada grupo de repartidor usando CVRP con OR-Tools,
    tiempos en segundos, manejo de 'inf' y cómputo de distancia recorrida por ruta.

    Args:
    df_comercios     : DataFrame con ['node', 'paquetes', 'geometry', 'grupo_repartidor'].
    capacidad_maxima : int, capacidad máxima de paquetes por ruta.

    Returns:
    routes_df        : DataFrame con rutas por cada grupo de repartidor ['grupo_repartidor', 'sequence', 'time_min', 'distance_km', 'load'].
    """

    all_routes = []
    total_time_sec = 0
    total_batches = 0
    INF_SEC = 10**9  # Valor para distancias infinitas (cuando no hay conexión)

    # Agrupar comercios por 'grupo_repartidor'
    grupos_repartidor = df_comercios["grupo_repartidor"].unique()

    # Iteramos sobre cada grupo de repartidor
    for grupo_sel in grupos_repartidor:
        # Filtrar comercios de ese grupo
        sub_bajos = df_comercios[df_comercios["grupo_repartidor"] == grupo_sel]
        N = len(sub_bajos) + 1

        # Si no hay comercios para este grupo, continuar con el siguiente
        if N == 0:
            continue

        # 1) Nodos y demandas (los comercios dentro del grupo + el hub)
        hub = hub_route["hub"]  # El nodo del hub
        locations = list(sub_bajos["node"])  # Los nodos de comercios
        locations.insert(0, hub)  # Añadir el hub como punto de inicio
        demands = list(sub_bajos["paquetes"])  # Demandas de paquetes
        demands.insert(0, 0)  # El hub tiene demanda cero

        # 2) Matriz de tiempos (segundos) entre los comercios
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
                    t_min = INF_SEC  # Si no hay ruta, marcarlo como "infinito"
                time_matrix[i][j] = (
                    INF_SEC if not np.isfinite(t_min) else int(math.ceil(t_min * 60))
                )  # Convertir a minutos

        # 3) Número de rutas necesarias
        total_demand = sum(demands)
        R = math.ceil(total_demand / capacidad_maxima)  # El número de rutas necesarias

        # 4) Configurar OR-Tools para el problema de CVRP
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
        search_params.time_limit.seconds = 15

        # Resolver el problema con OR-Tools
        try:
            sol = routing.SolveWithParameters(search_params)
            if not sol:
                raise RuntimeError(f"No solution CVRP for group {grupo_sel}")
        except RuntimeError:
            print(f"❌ Error: No se encontró solución para el grupo {grupo_sel}.")
            continue

        # 5) Extraer rutas y calcular distancia
        for v in range(R):
            idx = routing.Start(v)
            if routing.IsEnd(sol.Value(routing.NextVar(idx))):
                continue
            seq_nodes = []
            load = 0
            time_sec = 0
            dist_m = 0.0
            # Construir la ruta
            while not routing.IsEnd(idx):
                n = manager.IndexToNode(idx)
                seq_nodes.append(locations[n])
                load += demands[n]
                nxt = sol.Value(routing.NextVar(idx))
                time_sec += routing.GetArcCostForVehicle(idx, nxt, v)

                # Calcular la distancia tramo a tramo
                a, b = locations[n], locations[manager.IndexToNode(nxt)]
                _, path = calcular_tiempo_walk(G_super, a, b)  # Usamos G_super para la distancia
                if path:
                    for u, w in zip(path[:-1], path[1:]):
                        ed = G_super.get_edge_data(u, w)
                        dist_m += min(e.get("length", 0) for e in ed.values())

                idx = nxt

            # Retorno opcional al depósito (hub)
            seq_nodes.append(hub)

            # Almacenar resultados
            all_routes.append(
                {
                    "grupo_repartidor": grupo_sel,
                    "sequence": seq_nodes,
                    "time_min": round(time_sec / 60, 3),
                    "distance_km": round(dist_m / 1000, 3),
                    "load": load,
                }
            )
            total_time_sec += time_sec
            total_batches += 1

    # Crear un DataFrame con los resultados
    routes_df = pd.DataFrame(all_routes)
    return routes_df


def calculate_results_m2(routes_df: pd.DataFrame, hub_route: HubRoute) -> ModelResults:
    """
    Calcula el coste total del reparto sumando costes fijos, por tiempo y por distancia,
    diferenciando entre las rutas desde el **depósito al CID** (furgoneta) y las rutas
    desde el **CID al hub** (ONA).

    Args:
        routes_df    : DataFrame con ['time_min', 'distance_km', 'grupo_repartidor'] de las rutas generadas.
        hub_route   : Información del camino entre el depósito y el hub, incluyendo tiempo y distancia.

    Returns:
        total_cost   : Coste total calculado, que incluye los costes fijos, por tiempo y por distancia.
    """
    total_kms_walk = routes_df["distance_km"].sum()
    total_hours_walk = routes_df["time_min"].sum() / 60
    total_kms_drive = hub_route["dist"]
    total_hours_drive = hub_route["t_drive"] / 60

    distance_cost_van = total_kms_drive * DISTANCE_PER_KM_RATE_VAN
    distance_cost_ona = total_kms_walk * DISTANCE_PER_KM_RATE_ONA
    time_cost_van = total_hours_drive * TIME_HOURLY_RATE_VAN
    time_cost_ona = total_hours_walk * TIME_HOURLY_RATE_ONA

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


def map_m2(
    G_super: nx.MultiDiGraph,
    df_comercios: gpd.GeoDataFrame,
    hub_route: HubRoute,
    routes_df: pd.DataFrame,
    shp_zone: Polygon,
    min_dist_m=50,
) -> folium.Map:
    """
    Args:
    df_comercios   : DataFrame con ['node', 'paquetes', 'geometry', 'grupo_repartidor'].
    hub_route      : dict con información del hub y la ruta.
    routes_df      : DataFrame con las rutas generadas, con columnas ['grupo_repartidor', 'sequence', 'time_min', 'distance_km', 'load'].
    G_super        : Grafo de las calles, usado para obtener las coordenadas de los nodos.
    shp_zone       : Opción de polígono de zona (opcional).
    min_dist_m     : Distancia mínima en metros entre flechas en las rutas (opcional).
    view           : Booleano que indica si se debe mostrar el control de capas del mapa.

    Devuelve:
    folium.Map con:
    - Grafo de calles.
    - Comercios coloreados según 'grupo_repartidor'.
    - Hub como marcador.
    - Rutas de los grupos de repartidores con flechas.
    """
    # 1) Preparar paleta de colores (uno por cada grupo_repartidor)
    unique_groups = sorted(df_comercios["grupo_repartidor"].unique().tolist())
    valid_colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    random.seed(0)
    group_color_map = {
        group: valid_colors[i % len(valid_colors)] for i, group in enumerate(unique_groups)
    }

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

    # 5) Comercios coloreados según grupo_repartidor
    for _, row in df_comercios.iterrows():
        color = group_color_map[row["grupo_repartidor"]]
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

    # 6) Hub (marcador del hub)
    hubs_layer = folium.FeatureGroup(name="Hub Route", show=True)
    hub = hub_route["hub"]
    hub_x = G_super.nodes[hub]["x"]
    hub_y = G_super.nodes[hub]["y"]
    folium.Marker(
        location=(hub_y, hub_x),
        icon=folium.Icon(color="black", icon="warehouse", prefix="fa"),
        popup=f"Hub: {hub}",
    ).add_to(hubs_layer)

    camino_deposito_hub, t_drive, dist_drive = (
        hub_route["camino"],
        hub_route["t_drive"],
        hub_route["dist"],
    )
    if camino_deposito_hub:
        coords = [(G_super.nodes[n]["y"], G_super.nodes[n]["x"]) for n in camino_deposito_hub]
        folium.PolyLine(
            coords,
            color="black",
            weight=6,
            opacity=0.3,
            popup=f"Tiempo: {t_drive:.2f} min \nDistancia: {dist_drive:.2f} km",
        ).add_to(hubs_layer)

        # Añadir marcador para el primer nodo del camino (punto de entrada, depósito)
        entry_node = camino_deposito_hub[0]
        entry_x, entry_y = G_super.nodes[entry_node]["x"], G_super.nodes[entry_node]["y"]
        folium.Marker(
            location=(entry_y, entry_x),
            icon=folium.Icon(color="black", icon="play", prefix="fa"),
            popup=f"Punto de entrada: {entry_node}",
        ).add_to(hubs_layer)

        hubs_layer.add_to(m)

    # 7) Rutas de entrega (con flechas y detalles)
    route_layer = folium.FeatureGroup(name="Cluster Routes", show=True)
    for i, row in routes_df.iterrows():
        repartidor = row["grupo_repartidor"]
        color = valid_colors[i % len(valid_colors)]
        seq = row["sequence"]

        # reconstruir la lista de coordenadas de cada tramo
        coords = []
        for a, b in zip(seq[:-1], seq[1:]):
            _, path = calcular_tiempo_walk(G_super, a, b)
            if path:
                coords.extend([(G_super.nodes[n]["y"], G_super.nodes[n]["x"]) for n in path])

        # añadir la PolyLine punteada
        folium.PolyLine(
            coords,
            color=color,
            weight=3,
            opacity=0.5,
            dash_array="5,10",
            popup=f"Paquetes: {row['load']} \nTiempo: {row['time_min']:.2f} min \nDistancia: {row['distance_km']:.2f} km",
        ).add_to(route_layer)

        # calcular midpoints y colocar flechas
        mids = []
        for (lat1, lon1), (lat2, lon2) in zip(coords[:-1], coords[1:]):
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2
            dx, dy = lon2 - lon1, lat2 - lat1
            angle = math.degrees(math.atan2(dy, dx))
            mids.append({"pt": (mid_lat, mid_lon), "angle": -angle})

        # filtrar por distancia mínima entre flechas
        filtered = []
        last_pt = None
        for item in mids:
            if last_pt is None or haversine_dist(last_pt, item["pt"]) >= min_dist_m:
                filtered.append(item)
                last_pt = item["pt"]

        # colocar las flechas
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
                icon=folium.DivIcon(icon_size=(20, 20), icon_anchor=(10, 10), html=html),
            ).add_to(route_layer)

        # añadir los comercios en las rutas
        for nodo in seq:
            for _, row in df_comercios.iterrows():
                if nodo == row["node"]:
                    folium.CircleMarker(
                        location=(row.geometry.y, row.geometry.x),
                        radius=6,
                        color="black",
                        weight=0.5,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.9,
                        popup=f"{row['node']} \nPaquetes: {row['paquetes']}\nRuta: {i+1}",
                    ).add_to(route_layer)

    # 8) Incorporamos la capa y actualizamos control de capas
    route_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    return m


def simulation_M2(
    G_super: nx.MultiDiGraph,
    df_comercios: pd.DataFrame,
    nodo_entrada: int,
    hub: str,
    shp_zone: Polygon,
    capacidad_maxima=20,
) -> tuple[ModelResults, folium.Map]:
    hub_route = planificar_hub(G_super, nodo_entrada, hub)
    _, df_comercios = clustering(df_comercios, cap_max=capacidad_maxima)
    routes_df = cluster_routes(G_super, df_comercios, hub_route, capacidad_maxima)

    m = map_m2(G_super, df_comercios, hub_route, routes_df, shp_zone)
    results = calculate_results_m2(routes_df, hub_route)

    return results, m
