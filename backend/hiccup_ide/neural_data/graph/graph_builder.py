"""Graph building functionality for neural network coordinates.

This module provides functionality to build graphs by recursively finding parent
coordinates starting from a given coordinate, with optional filtering support.
"""

from typing import Callable, Set
import networkx as nx
from neural_data.types import Coordinate
from neural_data.model_spec import ModelDefinition
from neural_data.graph.parent_coordinates import (
    get_parents_of_conv2d_output_coordinate,
    get_parents_of_conv2d_slice_coordinate,
    get_parents_of_conv2d_input_coordinate,
    get_parents_of_relu_input_coordinate,
    get_parents_of_relu_output_coordinate,
    get_parents_of_model_input_coordinate,
)


# Mapping from coordinate type to parent-finding function
_COORDINATE_TO_PARENT_FUNC: dict[
    str, Callable[[Coordinate, ModelDefinition], list[Coordinate]]
] = {
    "Conv2dOutputCoordinate": get_parents_of_conv2d_output_coordinate,
    "Conv2dSliceCoordinate": get_parents_of_conv2d_slice_coordinate,
    "Conv2dInputCoordinate": get_parents_of_conv2d_input_coordinate,
    "ReLUInputCoordinate": get_parents_of_relu_input_coordinate,
    "ReLUOutputCoordinate": get_parents_of_relu_output_coordinate,
    "ModelInputCoordinate": get_parents_of_model_input_coordinate,
} # ty: ignore

FilterFunc = Callable[[Coordinate], bool]


def build_graph(
    start_coordinate: Coordinate,
    model_dfn: ModelDefinition,
    filter_func: FilterFunc = lambda _: True,
) -> nx.DiGraph:
    """Build a directed graph by recursively finding parent coordinates.

    The graph is built by successively finding parents of each coordinate until
    no more parents are found (typically at model input coordinates). Each
    coordinate is represented as a node in the graph, with directed edges
    pointing from child coordinates to their parent coordinates.

    Args:
        start_coordinate: The starting coordinate to build the graph from
        model_dfn: The model definition containing layer structure
        filter_func: Optional filter function that takes a coordinate and returns
                    whether it should be included in the graph. If a coordinate
                    is filtered out, that branch of graph building is terminated.
                    Defaults to always returning True (include all coordinates).

    Returns:
        NetworkX directed graph where:
        - Nodes are coordinate objects (used directly as node IDs since they are immutable)
        - Edges point from child coordinates to parent coordinates

    Raises:
        ValueError: If an unsupported coordinate type is encountered
    """
    graph = nx.DiGraph()
    visited: Set[Coordinate] = set()
    _build_recursive(
        start_coordinate, model_dfn, filter_func, visited, graph
    )
    return graph


def _build_recursive(
    coord: Coordinate, model_dfn: ModelDefinition, filter_func: FilterFunc, visited: set, graph: nx.Graph
) -> None:
    """Recursively build the graph starting from the given coordinate."""
    # Skip if already visited (prevents cycles)
    if coord in visited:
        return

    # Apply filter - if coordinate should not be included, terminate this branch
    if not filter_func(coord):
        return

    # Mark as visited and add to graph (using coordinate directly as node ID)
    visited.add(coord)
    graph.add_node(coord)

    # Get parent-finding function for this coordinate type
    if coord.type not in _COORDINATE_TO_PARENT_FUNC:
        raise ValueError(f"Unsupported coordinate type: {coord.type}")

    parent_func = _COORDINATE_TO_PARENT_FUNC[coord.type]

    # Find parents and recursively process them
    parents = parent_func(coord, model_dfn)
    for parent_coord in parents:
        # Recursively process parent
        _build_recursive(
            parent_coord, model_dfn, filter_func, visited, graph
        )

        # Add edge from current coordinate to parent (child -> parent direction)
        if parent_coord in graph:
            graph.add_edge(coord, parent_coord)
