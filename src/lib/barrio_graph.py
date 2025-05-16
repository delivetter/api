from networkx import MultiDiGraph


def get_nodos_carga(G: MultiDiGraph):
    return [n for n in G.nodes if G.nodes[n].get("tipo") == "carga"]


def get_nodos_comercios(G: MultiDiGraph):
    return [n for n in G.nodes if G.nodes[n].get("tipo") == "comercio"]
