"""Parent coordinate functions for Conv2D layers."""

from typing import List
from neural_data.types import (
    Conv2dOutputCoordinate,
    Conv2dSliceCoordinate,
    Conv2dInputCoordinate,
)
from neural_data.model_spec import ModelDefinition, LayerNode, Conv2dNode


def _find_layer_node(layer_name: str, model_dfn: ModelDefinition) -> LayerNode:
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


def get_parents_of_conv2d_output_coordinate(
    coord: Conv2dOutputCoordinate, model_dfn: ModelDefinition
) -> List[Conv2dSliceCoordinate]:
    """Get parent slice coordinates for a Conv2D output coordinate.

    For a Conv2D output coordinate, the parents are slice coordinates for each
    input channel at the same spatial position.

    Args:
        coord: The Conv2D output coordinate
        model_dfn: The model definition

    Returns:
        List of Conv2dSliceCoordinate objects, one for each input channel
    """
    layer_node = _find_layer_node(coord.layer_name, model_dfn)
    if not isinstance(layer_node, Conv2dNode):
        raise ValueError(f"Layer '{coord.layer_name}' is not a Conv2d layer")
    
    in_channels = layer_node.params.in_channels
    
    parents = []
    for in_channel in range(in_channels):
        parent = Conv2dSliceCoordinate(
            type="Conv2dSliceCoordinate",
            layer_name=coord.layer_name,
            layer_type="conv2d",
            coordinate_type="slice",
            in_channel=in_channel,
            out_channel=coord.channel,
            y=coord.y,
            x=coord.x,
        )
        parents.append(parent)
    return parents


def _get_receptive_field(
    output_y: int, 
    output_x: int, 
    kernel_size: int, 
    stride: int, 
    padding: int,
    input_height: int,
    input_width: int
) -> List[tuple[int, int]]:
    """Calculate the receptive field coordinates for a given output position.
    
    Args:
        output_y: Y coordinate in output space
        output_x: X coordinate in output space  
        kernel_size: Size of the convolution kernel (assuming square)
        stride: Stride of the convolution
        padding: Padding of the convolution
        input_height: Height of the input tensor
        input_width: Width of the input tensor
        
    Returns:
        List of (y, x) tuples representing valid input coordinates in the receptive field
    """
    receptive_field = []
    
    # Calculate the top-left corner of the receptive field in input space
    input_y_start = output_y * stride - padding
    input_x_start = output_x * stride - padding
    
    # Generate all coordinates in the kernel receptive field
    for ky in range(kernel_size):
        for kx in range(kernel_size):
            input_y = input_y_start + ky
            input_x = input_x_start + kx
            
            # Skip coordinates that are outside the valid input bounds
            if 0 <= input_y < input_height and 0 <= input_x < input_width:
                receptive_field.append((input_y, input_x))
    
    return receptive_field


def get_parents_of_conv2d_slice_coordinate(
    coord: Conv2dSliceCoordinate, model_dfn: ModelDefinition
) -> List[Conv2dInputCoordinate]:
    """Get parent input coordinates for a Conv2D slice coordinate.

    For a Conv2D slice coordinate, the parents are input coordinates that form
    the receptive field of that slice with the corresponding input channel.

    Args:
        coord: The Conv2D slice coordinate
        model_dfn: The model definition

    Returns:
        List of Conv2dInputCoordinate objects forming the receptive field
    """
    layer_node = _find_layer_node(coord.layer_name, model_dfn)
    if not isinstance(layer_node, Conv2dNode):
        raise ValueError(f"Layer '{coord.layer_name}' is not a Conv2d layer")
    
    params = layer_node.params
    
    kernel_size = params.kernel_size[0]  # Assuming square kernel
    stride = params.stride[0] 
    padding = params.padding[0]
    
    # Get input dimensions directly from the layer params
    input_shape = params.input_shape
    
    # Extract input dimensions (assuming shape is [batch, channels, height, width])
    input_height = input_shape[2]
    input_width = input_shape[3]
    
    # Get receptive field coordinates
    receptive_field_coords = _get_receptive_field(
        coord.y, coord.x, kernel_size, stride, padding, input_height, input_width
    )
    
    # Convert to Conv2dInputCoordinate objects
    parents = []
    for input_y, input_x in receptive_field_coords:
        parent = Conv2dInputCoordinate(
            type="Conv2dInputCoordinate",
            layer_name=coord.layer_name,
            layer_type="conv2d",
            coordinate_type="input",
            channel=coord.in_channel,
            y=input_y,
            x=input_x,
        )
        parents.append(parent)

    return parents
