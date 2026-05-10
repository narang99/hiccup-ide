"""Types for coordinating groups and pixels inside the network

NOTE: All types need to be consistently replicated in frontend at:
frontend/neural-viz/src/types/coordinates.ts
"""

from typing import Literal, Union, Annotated, assert_never
from pydantic import BaseModel, Field, ConfigDict


class ImmutableModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class Conv2dSliceGroup(ImmutableModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["Conv2dSliceGroup"]
    layer_name: str
    layer_type: Literal["conv2d"]
    coordinate_type: Literal["slice"]
    in_channel: int
    out_channel: int


class Conv2dSliceCoordinate(Conv2dSliceGroup):
    type: Literal["Conv2dSliceCoordinate"]
    y: int
    x: int


class ChannelGroup(ImmutableModel):
    layer_name: str
    channel: int


class Conv2dOutputGroup(ChannelGroup):
    type: Literal["Conv2dOutputGroup"]
    layer_type: Literal["conv2d"]
    coordinate_type: Literal["output"]


class Conv2dOutputCoordinate(Conv2dOutputGroup):
    type: Literal["Conv2dOutputCoordinate"]
    y: int
    x: int


class Conv2dInputGroup(ChannelGroup):
    type: Literal["Conv2dInputGroup"]
    layer_type: Literal["conv2d"]
    coordinate_type: Literal["input"]


class Conv2dInputCoordinate(Conv2dInputGroup):
    type: Literal["Conv2dInputCoordinate"]
    y: int
    x: int


class ReLUInputGroup(ChannelGroup):
    type: Literal["ReLUInputGroup"]
    layer_type: Literal["relu"]
    coordinate_type: Literal["input"]


class ReLUInputCoordinate(ReLUInputGroup):
    type: Literal["ReLUInputCoordinate"]
    y: int
    x: int


class ReLUOutputGroup(ChannelGroup):
    type: Literal["ReLUOutputGroup"]
    layer_type: Literal["relu"]
    coordinate_type: Literal["output"]


class ReLUOutputCoordinate(ReLUOutputGroup):
    type: Literal["ReLUOutputCoordinate"]
    y: int
    x: int


class ModelInputGroup(ChannelGroup):
    type: Literal["ModelInputGroup"]
    layer_type: Literal["input"]


class ModelInputCoordinate(ModelInputGroup):
    type: Literal["ModelInputCoordinate"]
    y: int
    x: int


TuplifiedInputCoordinates = tuple[tuple["Coordinate", tuple["Coordinate", ...]], ...]


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
    # we keep tuple -> its individual parents link also
    # this would be empty for now, just keeping it in the struct
    input_coordinates: TuplifiedInputCoordinates


class SingleConv2dInputPatch(ImmutableModel):
    """Data representing a patch of input coordinates for a single convolution operation."""

    layer_name: str
    layer_type: Literal["relu", "input"]
    coordinate_type: Literal["output_patch", "input_patch"]
    channel: int

    # Patch boundaries
    patch_min_y: int
    patch_min_x: int
    patch_max_y: int
    patch_max_x: int

    # References to the original coordinate objects
    input_coordinates: TuplifiedInputCoordinates


class SingleConv2dOpNode(ImmutableModel):
    """UI node representing a single channel convolution operation.

    This node consolidates multiple ReLUOutputCoordinate or ModelInputCoordinate
    objects that are dependencies of a Conv2dSliceCoordinate (via Conv2dInputCoordinates)
    into a single UI element.
    """

    type: Literal["SingleConv2dOpNode"]
    layer_name: str
    layer_type: Literal["conv2d"]
    coordinate_type: Literal["single_conv2d_op"]
    input_patch: SingleConv2dInputPatch
    output_slice: Conv2dSliceCoordinate


Coordinate = Annotated[
    Union[
        Conv2dInputCoordinate,
        Conv2dOutputCoordinate,
        Conv2dSliceCoordinate,
        ModelInputCoordinate,
        ReLUInputCoordinate,
        ReLUOutputCoordinate,
        Conv2dInputPatchNode,
        SingleConv2dOpNode,
    ],
    Field(discriminator="type"),
]

Group = Annotated[
    Union[
        Conv2dInputGroup,
        Conv2dSliceGroup,
        Conv2dOutputGroup,
        ModelInputGroup,
        ReLUInputGroup,
        ReLUOutputGroup,
    ],
    Field(discriminator="type"),
]


def to_coord_str(coord: Coordinate) -> str:
    match coord:
        case Conv2dInputCoordinate() as c:
            return f"{c.layer_name}.out_{c.channel}"
        case Conv2dOutputCoordinate() as c:
            return f"{c.layer_name}.out_{c.channel}"
        case Conv2dSliceCoordinate() as c:
            return f"{c.layer_name}.out_{c.out_channel}.in_{c.in_channel}"
        case ModelInputCoordinate() as c:
            return f"{c.layer_name}.out_{c.channel}"
        case ReLUInputCoordinate() as c:
            return f"{c.layer_name}.out_{c.channel}"
        case ReLUOutputCoordinate() as c:
            return f"{c.layer_name}.out_{c.channel}"
        case Conv2dInputPatchNode() as c:
            return f"{c.layer_name}.patch.in_{c.in_channel}.out_{c.out_channel}"
        case SingleConv2dOpNode() as c:
            return f"{c.layer_name}.op.{c.input_patch.layer_name}.ch_{c.input_patch.channel}"
        case _:
            assert_never(coord)


def to_coord_str_with_grid_position(coord: Coordinate) -> str:
    match coord:
        case Conv2dInputCoordinate() as c:
            return f"{c.layer_name}.in_{c.channel} ({c.y}, {c.x})"
        case Conv2dOutputCoordinate() as c:
            return f"{c.layer_name}.out_{c.channel} ({c.y}, {c.x})"
        case Conv2dSliceCoordinate() as c:
            return (
                f"{c.layer_name}.out_{c.out_channel}.in_{c.in_channel} ({c.y}, {c.x})"
            )
        case ModelInputCoordinate() as c:
            return f"{c.layer_name}.out_{c.channel} ({c.y}, {c.x})"
        case ReLUInputCoordinate() as c:
            return f"{c.layer_name}.in_{c.channel} ({c.y}, {c.x})"
        case ReLUOutputCoordinate() as c:
            return f"{c.layer_name}.out_{c.channel} ({c.y}, {c.x})"
        case Conv2dInputPatchNode() as c:
            return f"{c.layer_name}.patch.in_{c.in_channel}.out_{c.out_channel} ({c.patch_min_y}:{c.patch_max_y}, {c.patch_min_x}:{c.patch_max_x})"
        case SingleConv2dOpNode() as c:
            p = c.input_patch
            return f"{c.layer_name}.op.{p.layer_name}.ch_{p.channel} ({p.patch_min_y}:{p.patch_max_y}, {p.patch_min_x}:{p.patch_max_x})"
        case _:
            assert_never(coord)
