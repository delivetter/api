from typing import TypedDict


class ModelResults(TypedDict):
    total_kms_walk: float
    total_hours_walk: float
    total_kms_drive: float
    total_hours_drive: float
    distance_cost_van: float
    distance_cost_ona: float
    time_cost_van: float
    time_cost_ona: float
    total_cost: float
