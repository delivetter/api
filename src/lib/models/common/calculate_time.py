import networkx as nx
import math


def calcular_tiempo_walk(G: nx.DiGraph, origen: int, destino: int):
    """Ruta más rápida a pie entre origen y destino (minutos), usando el grafo G."""
    try:
        camino = nx.shortest_path(G, origen, destino, weight="tiempo_walk")
    except nx.NetworkXNoPath:
        return float("inf"), None
    tiempo = 0.0
    for u, v in zip(camino[:-1], camino[1:]):
        ed = G.get_edge_data(u, v)
        t_min = min(e["tiempo_walk"] for e in ed.values())
        tiempo += t_min
    return tiempo, camino


def calcular_tiempo_drive(G: nx.DiGraph, origen: int, destino: int):
    """Ruta más rápida en vehículo entre origen y destino (minutos), usando el grafo G."""
    try:
        camino = nx.shortest_path(G, origen, destino, weight="tiempo_drive")
    except nx.NetworkXNoPath:
        return float("inf"), []
    tiempo = 0.0
    for u, v in zip(camino[:-1], camino[1:]):
        ed = G.get_edge_data(u, v)
        tiempo += min(e.get("tiempo_drive", math.inf) for e in ed.values())
    return tiempo, camino
