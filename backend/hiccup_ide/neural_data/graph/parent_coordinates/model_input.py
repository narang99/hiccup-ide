"""Parent coordinate functions for model input layers."""

from typing import List
from neural_data.types import ModelInputCoordinate
from neural_data.model_spec import ModelDefinition


def get_parents_of_model_input_coordinate(coord: ModelInputCoordinate, model_dfn: ModelDefinition) -> List:
    """Get parent coordinates for a model input coordinate.
    
    Model input coordinates have no parents as they represent the initial input
    to the neural network.
    
    Args:
        coord: The model input coordinate
        model_dfn: The model definition (unused but kept for consistency)
        
    Returns:
        Empty list (no parents)
    """
    return []