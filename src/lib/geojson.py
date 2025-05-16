import json
import networkx as nx


def geojson_to_graph(path_in):
    """
    Lee un GeoJSON generado con graph_to_geojson y reconstruye
    un nx.MultiDiGraph con los mismos nodos, aristas y atributos.
    """
    with open(path_in, "r", encoding="utf-8") as f:
        gj = json.load(f)

    G = nx.MultiDiGraph()
    for feat in gj["features"]:
        geom = feat["geometry"]
        props = feat["properties"].copy()
        tipo = props.pop("_type")

        if tipo == "node":
            nid = props.pop("id")
            lon, lat = geom["coordinates"]
            # ahora guardamos pos, x y y
            G.add_node(nid, pos=(lat, lon), x=lon, y=lat, **props)

        elif tipo == "edge":
            u = props.pop("u")
            v = props.pop("v")
            key = props.pop("key", None)
            if u not in G:
                G.add_node(u)
            if v not in G:
                G.add_node(v)
            G.add_edge(u, v, key=key, **props)

    return G
