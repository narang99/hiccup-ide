"""Common utilities for parent coordinate functions."""

from typing import List, Callable, Any
from functools import partial
from neural_data.types import (
    Coordinate,
    Conv2dInputCoordinate,
    Conv2dOutputCoordinate,
    ReLUInputCoordinate,
    ReLUOutputCoordinate,
    ModelInputCoordinate,
)
from neural_data.model_spec import ModelDefinition, LayerNode


def find_layer_node(layer_name: str, model_dfn: ModelDefinition) -> LayerNode:
    """Find a layer node in the model definition by its ID.
    
    Args:
        layer_name: The ID of the layer to find
        model_dfn: The model definition
        
    Returns:
        The layer node
        
    Raises:
        ValueError: If the layer is not found
    """
    for node in model_dfn.nodes:
        if node.id == layer_name:
            return node
    raise ValueError(f"Layer '{layer_name}' not found in model definition")


def find_previous_layer(layer_name: str, model_dfn: ModelDefinition) -> str:
    """Find the previous layer that feeds into this layer.
    
    Args:
        layer_name: The current layer name
        model_dfn: The model definition
        
    Returns:
        The name of the previous layer
        
    Raises:
        ValueError: If no previous layer is found
    """
    for edge in model_dfn.edges:
        if edge.target == layer_name:
            return edge.source
    raise ValueError(f"No previous layer found for '{layer_name}'")


def _create_conv2d_output_coordinate(layer_name: str, channel: int, y: int, x: int) -> Conv2dOutputCoordinate:
    """Create a Conv2d output coordinate."""
    return Conv2dOutputCoordinate(
        type="Conv2dOutputCoordinate",
        layer_name=layer_name,
        layer_type="conv2d",
        coordinate_type="output",
        channel=channel,
        y=y,
        x=x
    )


def _create_relu_output_coordinate(layer_name: str, channel: int, y: int, x: int) -> ReLUOutputCoordinate:
    """Create a ReLU output coordinate."""
    return ReLUOutputCoordinate(
        type="ReLUOutputCoordinate",
        layer_name=layer_name,
        layer_type="relu",
        coordinate_type="output",
        channel=channel,
        y=y,
        x=x
    )


def _create_model_input_coordinate(layer_name: str, channel: int, y: int, x: int) -> ModelInputCoordinate:
    """Create a model input coordinate."""
    return ModelInputCoordinate(
        type="ModelInputCoordinate",
        layer_name=layer_name,
        layer_type="input",
        channel=channel,
        y=y,
        x=x
    )


# Mapping from layer type to coordinate creation function
_LAYER_TYPE_TO_OUTPUT_CREATOR = {
    "Conv2d": _create_conv2d_output_coordinate,
    "ReLU": _create_relu_output_coordinate,
    "Input": _create_model_input_coordinate,
}


def get_parents_for_input_coordinate(
    coord, 
    model_dfn: ModelDefinition,
    coordinate_extractor: Callable[[Any], tuple[int, int, int]]
) -> List[Coordinate]:
    """Generic function to get parent coordinates for any input coordinate.
    
    This function handles the common logic of:
    1. Finding the previous layer
    2. Determining the output coordinate type for that layer
    3. Creating the appropriate output coordinate with same spatial/channel position
    
    Args:
        coord: The input coordinate (any type with coordinate_type="input")
        model_dfn: The model definition
        coordinate_extractor: Function that extracts (channel, y, x) from the coordinate
        
    Returns:
        List containing the output coordinate from the previous layer
        
    Raises:
        ValueError: If the previous layer type is not supported
    """
    # Extract coordinate components
    channel, y, x = coordinate_extractor(coord)
    
    # Find previous layer
    previous_layer_name = find_previous_layer(coord.layer_name, model_dfn)
    previous_layer_node = find_layer_node(previous_layer_name, model_dfn)
    
    # Get the appropriate coordinate creator for the previous layer type
    if previous_layer_node.type not in _LAYER_TYPE_TO_OUTPUT_CREATOR:
        raise ValueError(f"Unsupported previous layer type: {previous_layer_node.type}")
    
    creator_func = _LAYER_TYPE_TO_OUTPUT_CREATOR[previous_layer_node.type]
    
    # Create the parent coordinate
    parent = creator_func(previous_layer_name, channel, y, x)
    
    return [parent]


# Pre-configured functions for specific coordinate types using partial application
def _extract_conv2d_input_coords(coord: Conv2dInputCoordinate) -> tuple[int, int, int]:
    """Extract (channel, y, x) from Conv2dInputCoordinate."""
    return coord.channel, coord.y, coord.x


def _extract_relu_input_coords(coord: ReLUInputCoordinate) -> tuple[int, int, int]:
    """Extract (channel, y, x) from ReLUInputCoordinate."""
    return coord.channel, coord.y, coord.x


# Configured functions for each coordinate type
get_parents_for_conv2d_input = partial(
    get_parents_for_input_coordinate,
    coordinate_extractor=_extract_conv2d_input_coords
)

get_parents_for_relu_input = partial(
    get_parents_for_input_coordinate,
    coordinate_extractor=_extract_relu_input_coords
)