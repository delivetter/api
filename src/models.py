from pydantic import BaseModel


class Input(BaseModel):
    barrio: str
    num_paquetes: int
    nodo_entrada: int
    almacen: str


class ModelResults(BaseModel):
    total_kms_walk: float
    total_hours_walk: float
    total_kms_drive: float
    total_hours_drive: float
    distance_cost_van: float
    distance_cost_ona: float
    time_cost_van: float
    time_cost_ona: float
    total_cost: float


class ModelOutput(BaseModel):
    results: ModelResults
    map_html: str


class SimulateOutput(BaseModel):
    m1: ModelOutput
    m2: ModelOutput
