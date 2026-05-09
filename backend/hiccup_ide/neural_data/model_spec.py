"""Type definitions for neural network model specifications.

This module provides complete type definitions for the model definition JSON structure
using discriminated unions similar to the coordinate types in types.py.
"""

from typing import Literal, Union, Annotated, Any
from pydantic import BaseModel, Field, ConfigDict


class ImmutableModel(BaseModel):
    """Base class for immutable model specifications."""

    model_config = ConfigDict(frozen=True)


class BaseNode(ImmutableModel):
    id: str
    shape: list[int]  # backwards compat, not useful


class InputParams(ImmutableModel):
    output_shape: list[int]  # Output tensor shape (same as input shape)


class InputNode(BaseNode):
    type: Literal["Input"]
    params: InputParams


class Conv2dParams(ImmutableModel):
    in_channels: int
    out_channels: int
    kernel_size: list[int]  # [height, width]
    stride: list[int]  # [height, width]
    padding: list[int]  # [height, width]
    input_shape: list[int]  # Input tensor shape [batch, channels, height, width]
    output_shape: list[int]  # Output tensor shape [batch, channels, height, width]


class Conv2dNode(BaseNode):
    type: Literal["Conv2d"]
    params: Conv2dParams


class ReLUParams(ImmutableModel):
    input_shape: list[int]  # Input tensor shape
    output_shape: list[int]  # Output tensor shape (same as input for ReLU)


class ReLUNode(BaseNode):
    type: Literal["ReLU"]
    params: ReLUParams


class FlattenParams(ImmutableModel):
    input_shape: list[
        int
    ]  # Input tensor shape (e.g., [batch, channels, height, width])
    output_shape: list[int]  # Output tensor shape (e.g., [batch, flattened_size])


class FlattenNode(BaseNode):
    type: Literal["Flatten"]
    params: FlattenParams


class LinearParams(ImmutableModel):
    in_features: int
    out_features: int
    input_shape: list[int]  # Input tensor shape
    output_shape: list[int]  # Output tensor shape


class LinearNode(BaseNode):
    type: Literal["Linear"]
    params: LinearParams


class OutputParams(ImmutableModel):
    input_shape: list[int]  # Input tensor shape


class OutputNode(BaseNode):
    type: Literal["Output"]
    params: OutputParams


class ModelEdge(ImmutableModel):
    source: str
    target: str


# Discriminated union of all layer node types
LayerNode = Annotated[
    Union[
        InputNode,
        Conv2dNode,
        ReLUNode,
        FlattenNode,
        LinearNode,
        OutputNode,
    ],
    Field(discriminator="type"),
]


class ModelDefinition(ImmutableModel):
    """Complete model definition specification."""

    nodes: list[LayerNode]
    edges: list[ModelEdge]
