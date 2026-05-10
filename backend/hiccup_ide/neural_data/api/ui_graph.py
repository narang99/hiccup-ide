from typing import List
from ninja import Router, Schema
from django.shortcuts import get_object_or_404
import networkx as nx

from functools import partial
from neural_data.models import Model, Work, WorkGraph, WorkSaliencyMap
from neural_data.types import (
    Coordinate, 
    Conv2dInputCoordinate,
    ReLUInputCoordinate,
    ModelInputCoordinate,
    to_coord_str
)
from neural_data.graph.raw import build_graph
from neural_data.graph.ui_tfm import raw_to_ui_graph
from neural_data.model_spec import ModelDefinition

router = Router()

class NodeSchema(Schema):
    id: Coordinate

class EdgeSchema(Schema):
    source: Coordinate
    target: Coordinate

class UIGraphSchema(Schema):
    nodes: List[NodeSchema]
    edges: List[EdgeSchema]
    directed: bool
    multigraph: bool

def filter_func(coord: Coordinate, work_graph: WorkGraph):
    # only allow pos saliency map coords
    # Input coordinates are treated as "pass-through" 
    # for the purpose of saliency filtering (they don't have their own saliency maps usually)
    if isinstance(coord, (Conv2dInputCoordinate, ReLUInputCoordinate, ModelInputCoordinate)):
        return True
    
    coord_str = to_coord_str(coord)
    try:
        sm = WorkSaliencyMap.objects.get(coordinate=coord_str, graph=work_graph)
        # Check if the value at the specific grid position is positive
        # we know it works because we only allow the coordinate types which have these
        assert hasattr(coord, "y")
        assert hasattr(coord, "x")
        return sm.data[coord.y][coord.x] > 0
    except WorkSaliencyMap.DoesNotExist:
        # If we can't find a saliency map for a coordinate that isn't Input, 
        # it might be an issue with data loading or pruning state
        raise Exception(f"Work saliency map does not exist for coordinate, query={coord_str} coord={coord}")

@router.post(
    "/models/{model_alias}/inputs/{input_alias}/workflows/{work_alias}/ui-graph/",
    response=UIGraphSchema
)
def get_ui_graph(
    request, 
    model_alias: str, 
    input_alias: str, 
    work_alias: str, 
    payload: List[Coordinate]
):
    model = get_object_or_404(Model, alias=model_alias)
    model_dfn = ModelDefinition.model_validate(model.definition)
    
    # Get the specific work graph to access its saliency maps
    work = get_object_or_404(Work, input__model=model, input__alias=input_alias, name=work_alias)
    work_graph = get_object_or_404(WorkGraph, work=work)
    
    # Prepare the filter function with partial application
    bound_filter = partial(filter_func, work_graph=work_graph)
    
    # Build raw graph from all coordinates provided, applying the saliency filter
    full_raw_graph = nx.DiGraph()
    for coord in payload:
        g = build_graph(coord, model_dfn, filter_func=bound_filter)
        full_raw_graph = nx.compose(full_raw_graph, g)
            
    # Transform to UI graph
    ui_graph = raw_to_ui_graph(full_raw_graph)
    
    # Convert to node-link format for JSON response
    # NetworkX node_link_data will use the Coordinate objects as IDs
    data = nx.node_link_data(ui_graph, edges="edges")
    
    return data
