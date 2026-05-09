"""Parent coordinate functions for ReLU layers."""

from typing import List
from neural_data.types import (
    ReLUInputCoordinate,
    ReLUOutputCoordinate,
    Conv2dOutputCoordinate,
)
from neural_data.model_spec import ModelDefinition


def _find_previous_layer(layer_name: str, model_dfn: ModelDefinition) -> str:
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


def get_parents_of_relu_input_coordinate(coord: ReLUInputCoordinate, model_dfn: ModelDefinition) -> List[Conv2dOutputCoordinate]:
    """Get parent coordinates for a ReLU input coordinate.
    
    For a ReLU input coordinate, the parent is the Conv2D output coordinate
    at the same spatial position and channel from the previous layer.
    
    Args:
        coord: The ReLU input coordinate
        model_dfn: The model definition
        
    Returns:
        List containing a single Conv2dOutputCoordinate
    """
    previous_layer_name = _find_previous_layer(coord.layer_name, model_dfn)
    
    parent = Conv2dOutputCoordinate(
        type="Conv2dOutputCoordinate",
        layer_name=previous_layer_name,
        layer_type="conv2d",
        coordinate_type="output",
        channel=coord.channel,
        y=coord.y,
        x=coord.x
    )
    return [parent]


def get_parents_of_relu_output_coordinate(coord: ReLUOutputCoordinate, model_dfn: ModelDefinition) -> List[ReLUInputCoordinate]:
    """Get parent coordinates for a ReLU output coordinate.
    
    For a ReLU output coordinate, the parent is the ReLU input coordinate
    at the same spatial position and channel (ReLU is element-wise).
    
    Args:
        coord: The ReLU output coordinate
        model_dfn: The model definition
        
    Returns:
        List containing a single ReLUInputCoordinate
    """
    parent = ReLUInputCoordinate(
        type="ReLUInputCoordinate",
        layer_name=coord.layer_name,
        layer_type="relu",
        coordinate_type="input",
        channel=coord.channel,
        y=coord.y,
        x=coord.x
    )
    return [parent]