"""UI graph builder for neural network coordinate visualization.

This module provides the main entry point for building UI-optimized graphs
by combining raw graph construction with UI-specific transformations.
"""
import networkx as nx
from neural_data.types import Coordinate
from neural_data.model_spec import ModelDefinition
from neural_data.graph.graph_builder import build_graph, FilterFunc
from neural_data.graph.ui_graph_transformer import transform_raw_graph_to_ui_graph


def build_ui_graph(
    start_coordinate: Coordinate,
    model_dfn: ModelDefinition,
    filter_func: FilterFunc = lambda _: True,
) -> nx.DiGraph:
    """Build a UI-optimized graph from a starting coordinate.
    
    This function creates a dependency graph optimized for UI visualization by:
    1. Building the raw computational dependency graph
    2. Applying UI-specific transformations (e.g., consolidating receptive fields)
    
    Args:
        start_coordinate: The starting coordinate to build the graph from
        model_dfn: The model definition containing layer structure
        filter_func: Optional filter function for including/excluding coordinates.
                    Applied to raw coordinates before transformation.
                    
    Returns:
        NetworkX directed graph with UI-optimized node types where:
        - Nodes are UI graph node objects (Conv2dInputPatchNode, etc.)
        - Edges point from parent to child nodes
        - Conv2dSliceCoordinate -> [Conv2dInputCoordinate] relationships are
          consolidated into single Conv2dInputPatchNode objects
          
    Raises:
        ValueError: If an unsupported coordinate type is encountered
    """
    # Build the raw computational graph
    raw_graph = build_graph(start_coordinate, model_dfn, filter_func)
    
    # Transform to UI-optimized representation  
    ui_graph = transform_raw_graph_to_ui_graph(raw_graph)
    
    return ui_graph