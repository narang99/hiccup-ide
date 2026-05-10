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


Coordinate = Annotated[
    Union[
        Conv2dInputCoordinate,
        Conv2dOutputCoordinate,
        Conv2dSliceCoordinate,
        ModelInputCoordinate,
        ReLUInputCoordinate,
        ReLUOutputCoordinate,
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
            return (
                f"{c.layer_name}.out_{c.out_channel}.in_{c.in_channel}"
            )
        case ModelInputCoordinate() as c:
            return f"{c.layer_name}.out_{c.channel}"
        case ReLUInputCoordinate() as c:
            return f"{c.layer_name}.out_{c.channel}"
        case ReLUOutputCoordinate() as c:
            return f"{c.layer_name}.out_{c.channel}"
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
        case _:
            assert_never(coord)
