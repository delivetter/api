from lib.barrio_graph import get_nodos_comercios
import numpy as np
import networkx as nx
import geopandas as gpd
from shapely import Point


def generate_context(
    G_super: nx.DiGraph,
    n_paquetes: int,
    paquetes_min: int = 1,
    paquetes_max: int = 5,
    capacidad_maxima: int = 8,
    factor_ajuste: float = 0.75,
):
    # Instanciamos un RNG local
    rng = np.random.RandomState()

    # Validación de parámetros
    if (
        paquetes_max < paquetes_min
        or paquetes_min < 0
        or paquetes_max < 0
        or capacidad_maxima < 0
        or capacidad_maxima < min(paquetes_min, paquetes_max)
    ):
        raise ValueError("Invalid values for constants. Check the input values.")

    nodos_comercios = get_nodos_comercios(G_super)
    n_comercios = len(nodos_comercios)
    max_paquetes = n_comercios * capacidad_maxima
    proporcion = n_paquetes / max_paquetes

    # Ajuste preliminar de N_PAQUETES si la proporción es alta
    if proporcion >= factor_ajuste:
        n_paquetes = int(max_paquetes * factor_ajuste)

    # Inicializar capacidades y lista de nodos disponibles
    capacidades = {nodo: capacidad_maxima for nodo in nodos_comercios}
    available_nodes = nodos_comercios.copy()  # lista mutable de nodos disponibles

    # Listas para almacenar asignaciones
    seleccionados = []
    paquetes_asignados_list = []

    paquetes_asignados = 0
    n_paq = n_paquetes

    # Bucle principal: mientras haya paquetes por asignar y nodos disponibles
    while paquetes_asignados < n_paquetes and available_nodes:
        restante = n_paquetes - paquetes_asignados
        max_pos = min(paquetes_max, restante)
        if max_pos < paquetes_min:
            break

        # Selección aleatoria eficiente:
        idx = available_nodes.index(rng.choice(available_nodes))
        nodo = available_nodes[idx]
        cap_restante = capacidades[nodo]

        # Si el nodo no tiene capacidad para PAQUETES_MIN, se elimina de la lista
        if cap_restante < paquetes_min:
            available_nodes[idx] = available_nodes[-1]
            available_nodes.pop()
            continue

        # Determinar la cantidad de paquetes a asignar
        paquetes_a_asignar = rng.randint(paquetes_min, min(max_pos, cap_restante) + 1)

        # Registrar la asignación
        seleccionados.append(nodo)
        paquetes_asignados_list.append(paquetes_a_asignar)

        capacidades[nodo] -= paquetes_a_asignar
        paquetes_asignados += paquetes_a_asignar

        # Si tras la asignación la capacidad es insuficiente, se elimina el nodo de la lista
        if capacidades[nodo] < paquetes_min:
            available_nodes[idx] = available_nodes[-1]
            available_nodes.pop()

    # Agrupar las asignaciones por nodo
    asignacion_por_nodo = {}
    for nodo, paquetes in zip(seleccionados, paquetes_asignados_list):
        asignacion_por_nodo[nodo] = asignacion_por_nodo.get(nodo, 0) + paquetes

    # Construir la lista de datos para el GeoDataFrame usando la variable global G_super para obtener coordenadas
    data = []
    for nodo, total in asignacion_por_nodo.items():
        lon = G_super.nodes[nodo]["x"]
        lat = G_super.nodes[nodo]["y"]
        data.append({"node": nodo, "paquetes": total, "geometry": Point(lon, lat)})

    return gpd.GeoDataFrame(data, crs="EPSG:4326")
