from ninja import Schema
from typing import List, Any, Optional


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


class WorkTreeOut(Schema):
    alias: str
    name: str

class WorkIn(Schema):
    name: str

class InputTreeOut(Schema):
    alias: str
    name: str
    works: List[WorkTreeOut]

class ModelTreeOut(Schema):
    alias: str
    name: str
    inputs: List[InputTreeOut]

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

class WorkGraphMeta(Schema):
    work_alias: str



class ActivationOut(Schema):
    id: int
    coordinate: str
    data: Any
    shape: List[int]
    layer_type: str
    coordinate_type: str
    output_channel: Optional[int] = None
    input_channel: Optional[int] = None
    work_graph: Optional[WorkGraphMeta] = None


class SaliencyMapOut(Schema):
    id: int
    coordinate: str
    layer_name: str
    data: Any
    shape: List[int]
    coordinate_type: str
    data_type: str
    output_channel: Optional[int] = None
    input_channel: Optional[int] = None
    work_graph: Optional[WorkGraphMeta] = None


class SaliencyMapListOut(Schema):
    items: List[SaliencyMapOut]


class WeightOut(Schema):
    id: int
    coordinate: str
    data: Any
    shape: List[int]
    layer_type: str
    coordinate_type: str
    data_type: str
    output_channel: Optional[int] = None
    input_channel: Optional[int] = None
    work_graph: Optional[WorkGraphMeta] = None


class BatchSaliencyMapsIn(Schema):
    coordinates: List[str]


class NodeStatsOut(Schema):
    min: float
    max: float


class ThresholdAlgorithm(Schema):
    type: str = "ThresholdAlgorithm"
    threshold: float


class IdAlgorithm(Schema):
    type: str = "Id"


class CoordinateAlgorithmIn(Schema):
    coordinate: str
    algorithm: dict  # Using dict for flexible discriminated union in Ninja


class BatchWorkSaliencyMapIn(Schema):
    items: List[CoordinateAlgorithmIn]


class PruningStatusLayers(Schema):
    done: List[str]
    total: List[str]


class PruningStatusOut(Schema):
    layers: PruningStatusLayers
    session_active: bool


class POIIn(Schema):
    work_alias: str
    weight_coordinate: str
    x: int
    y: int
    label: str
    note: str


class POIOut(Schema):
    id: int
    work_alias: str
    weight_coordinate: str
    x: int
    y: int
    label: str
    note: str


class InputLayerMeta(Schema):
    type: str
    layer_name: str


class POIPoint(Schema):
    row: int
    col: int
    value: float


class UniqueActivationId(Schema):
    input_alias: str
    model_alias: str
    coordinate: str


class HighActivatedPOIOut(Schema):
    output_activation: UniqueActivationId
    input_activations: list[UniqueActivationId]
    point: POIPoint


class HighActivatedPOIsResponse(Schema):
    pois: List[HighActivatedPOIOut]


class KernelLabelsIn(Schema):
    weight_coordinate: str
    labels: List[str]


class KernelLabelsOut(Schema):
    id: int
    weight_coordinate: str
    labels: List[str]
    created_at: str
    updated_at: str


class MarkSliceDoneIn(Schema):
    coordinate: str
