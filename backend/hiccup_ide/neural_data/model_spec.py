"""Type definitions for neural network model specifications.

This module provides complete type definitions for the model definition JSON structure
using discriminated unions similar to the coordinate types in types.py.
"""

from typing import Literal, Union, Annotated, Any
from pydantic import BaseModel, Field, ConfigDict


class ImmutableModel(BaseModel):
    """Base class for immutable model specifications."""

    model_config = ConfigDict(frozen=True)


class InputParams(ImmutableModel):
    """Parameters for Input layers."""

    output_shape: list[int]  # Output tensor shape (same as input shape)


class InputNode(ImmutableModel):
    """Input layer node specification."""

    id: str
    type: Literal["Input"]
    params: InputParams
    shape: list[int]  # Output shape (same as input shape for input nodes)


class Conv2dParams(ImmutableModel):
    """Parameters for Conv2d layers."""

    in_channels: int
    out_channels: int
    kernel_size: list[int]  # [height, width]
    stride: list[int]  # [height, width]
    padding: list[int]  # [height, width]
    input_shape: list[int]  # Input tensor shape [batch, channels, height, width]
    output_shape: list[int]  # Output tensor shape [batch, channels, height, width]


class Conv2dNode(ImmutableModel):
    """Conv2d layer node specification."""

    id: str
    type: Literal["Conv2d"]
    params: Conv2dParams
    shape: list[int]  # Output shape (kept for backward compatibility)


class ReLUParams(ImmutableModel):
    """Parameters for ReLU layers."""

    input_shape: list[int]  # Input tensor shape
    output_shape: list[int]  # Output tensor shape (same as input for ReLU)


class ReLUNode(ImmutableModel):
    """ReLU layer node specification."""

    id: str
    type: Literal["ReLU"]
    params: ReLUParams
    shape: list[int]


class FlattenParams(ImmutableModel):
    """Parameters for Flatten layers."""

    input_shape: list[int]  # Input tensor shape (e.g., [batch, channels, height, width])
    output_shape: list[int]  # Output tensor shape (e.g., [batch, flattened_size])


class FlattenNode(ImmutableModel):
    """Flatten layer node specification."""

    id: str
    type: Literal["Flatten"]
    params: FlattenParams
    shape: list[int]


class LinearParams(ImmutableModel):
    """Parameters for Linear layers."""

    in_features: int
    out_features: int
    input_shape: list[int]  # Input tensor shape
    output_shape: list[int]  # Output tensor shape


class LinearNode(ImmutableModel):
    """Linear layer node specification."""

    id: str
    type: Literal["Linear"]
    params: LinearParams
    shape: list[int]


class OutputParams(ImmutableModel):
    """Parameters for Output layers."""

    input_shape: list[int]  # Input tensor shape


class OutputNode(ImmutableModel):
    """Output layer node specification."""

    id: str
    type: Literal["Output"]
    params: OutputParams
    shape: list[int]


class ModelEdge(ImmutableModel):
    """Edge connecting two layers in the model."""

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

