"""UI-specific graph types for neural network coordinate visualization.

This module defines types specifically designed for UI consumption, transforming
the raw computational graph into a form optimized for visualization.

NOTE: All types need to be consistently replicated in frontend at:
frontend/neural-viz/src/types/ui_graph_coordinates.ts
"""
from typing import Literal, Union, Annotated
from pydantic import BaseModel, Field, ConfigDict
from neural_data.types import (
    Conv2dInputCoordinate, 
    Conv2dOutputCoordinate,
    Conv2dSliceCoordinate,
    ReLUInputCoordinate,
    ReLUOutputCoordinate, 
    ModelInputCoordinate,
    ImmutableModel,
)


class Conv2dInputPatchNode(ImmutableModel):
    """UI node representing a patch of input coordinates for Conv2d visualization.
    
    This node consolidates multiple Conv2dInputCoordinate objects that form
    the receptive field of a Conv2dSliceCoordinate into a single UI element
    that can display the input patch with highlighted regions.
    """
    type: Literal["Conv2dInputPatchNode"]
    layer_name: str
    layer_type: Literal["conv2d"] 
    coordinate_type: Literal["input_patch"]
    
    # The input channel this patch represents
    in_channel: int
    
    # The output channel this patch connects to
    out_channel: int
    
    # Patch boundaries (min/max coordinates of the receptive field)
    patch_min_y: int
    patch_min_x: int
    patch_max_y: int
    patch_max_x: int
    
    # References to the original Conv2dInputCoordinate objects
    # that form this patch (for detailed inspection if needed)
    # Note: Using tuple instead of list to maintain hashability for NetworkX
    input_coordinates: tuple[Conv2dInputCoordinate, ...]


# UI Graph Node Types - Union of all node types used in UI graphs
UIGraphNode = Annotated[
    Union[
        # Transformed types
        Conv2dInputPatchNode,
        # Pass-through types from raw graph (unchanged for UI)
        Conv2dOutputCoordinate,  # Used as-is
        ReLUInputCoordinate,     # Used as-is
        ModelInputCoordinate,    # Used as-is
        # Note: These types are NOT included in UI graphs:
        # - Conv2dInputCoordinate: transformed into Conv2dInputPatchNode
        # - Conv2dSliceCoordinate: always transformed into Conv2dInputPatchNode
        # - ReLUOutputCoordinate: removed to simplify graph
    ],
    Field(discriminator="type"),
]