"""Parent coordinate functions for ReLU layers."""

from neural_data.types import (
    ReLUInputCoordinate,
    ReLUOutputCoordinate,
)
from neural_data.model_spec import ModelDefinition
from neural_data.graph.parent_coordinates.utils import get_parents_for_input_coordinate


def get_parents_of_relu_input_coordinate(
    coord: ReLUInputCoordinate, model_dfn: ModelDefinition
) -> list:
    """Get parent coordinates for a ReLU input coordinate.

    For a ReLU input coordinate, the parent is the output coordinate
    from the previous layer at the same spatial position and channel.

    Args:
        coord: The ReLU input coordinate
        model_dfn: The model definition

    Returns:
        List containing the output coordinate from the previous layer
    """

    def _extract_relu_input_coords(coord: ReLUInputCoordinate) -> tuple[int, int, int]:
        """Extract (channel, y, x) from ReLUInputCoordinate."""
        return coord.channel, coord.y, coord.x

    return get_parents_for_input_coordinate(
        coord, model_dfn, _extract_relu_input_coords
    )


def get_parents_of_relu_output_coordinate(
    coord: ReLUOutputCoordinate, model_dfn: ModelDefinition
) -> list[ReLUInputCoordinate]:
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
        x=coord.x,
    )
    return [parent]
