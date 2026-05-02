from ninja import Schema
from typing import List, Any


class ModelNode(Schema):
    id: str
    type: str
    params: dict
    shape: List[int]


class ModelEdge(Schema):
    source: str
    target: str


class ModelDefinition(Schema):
    nodes: List[ModelNode]
    edges: List[ModelEdge]


class ModelOut(Schema):
    id: int
    alias: str
    name: str
    definition: ModelDefinition


class InputOut(Schema):
    id: int
    alias: str
    name: str
    data_path: str


class ActivationOut(Schema):
    id: int
    coordinate: str
    data: Any
    shape: List[int]
    layer_type: str
    coordinate_type: str


class SaliencyMapOut(Schema):
    id: int
    coordinate: str
    data: Any
    shape: List[int]
    coordinate_type: str
    data_type: str


class WeightOut(Schema):
    id: int
    coordinate: str
    data: Any
    shape: List[int]
    layer_type: str
    coordinate_type: str
    data_type: str


class LayerSaliencyMapsOut(Schema):
    layer_name: str
    saliency_maps: List[SaliencyMapOut]


class BatchSaliencyMapsIn(Schema):
    coordinates: List[str]
